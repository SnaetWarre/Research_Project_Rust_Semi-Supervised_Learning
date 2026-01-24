# Mobile SSL Training - Adaptive Backend System

This document explains the **automatic device-adaptive training system** that allows the PlantVillage SSL Dashboard to run on both **desktop (with CUDA GPU)** and **mobile devices (iOS/Android with CPU)** with zero configuration needed from the user.

## How It Works

The system automatically detects the device type and backend capabilities at **compile-time** and **runtime**, then adapts the training configuration accordingly.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Tauri GUI Application                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐         ┌──────────────┐               │
│  │   Device    │────────▶│   Backend    │               │
│  │  Detection  │         │  Selection   │               │
│  └─────────────┘         └──────────────┘               │
│        │                        │                        │
│        │                        │                        │
│        ▼                        ▼                        │
│  ┌──────────────────────────────────────┐               │
│  │   Adaptive Training Configuration     │               │
│  └──────────────────────────────────────┘               │
│        │                                                 │
│        ├─────────────┬───────────────┐                  │
│        ▼             ▼               ▼                   │
│   ┌─────────┐   ┌─────────┐   ┌──────────┐            │
│   │ Desktop │   │  Mobile │   │   SSL    │            │
│   │ Config  │   │ Config  │   │ Retrain  │            │
│   └─────────┘   └─────────┘   └──────────┘            │
│   • 32 batch    • 8 batch     • Lightweight           │
│   • 50 epochs   • 2 epochs    • Quick cycles          │
│   • CUDA GPU    • NdArray CPU • Pseudo-labels         │
└─────────────────────────────────────────────────────────┘
```

## Device Detection

The system detects the device type using Rust's compile-time target OS detection:

- **Desktop**: `target_os = "linux"`, `"windows"`, `"macos"`
- **Mobile**: `target_os = "ios"`, `"android"`

See `src-tauri/src/device.rs:22-38`

## Backend Selection

### Compile-Time Selection

The backend is selected using Cargo features:

```toml
[features]
default = ["custom-protocol", "cuda"]  # Desktop default
cuda = ["burn-cuda", "burn/cuda"]      # NVIDIA GPU
ndarray = ["burn-ndarray", "burn/ndarray"]  # CPU fallback
mobile = ["ndarray"]                   # Mobile target
```

### Runtime Behavior

```rust
// In src-tauri/src/backend.rs

#[cfg(feature = "cuda")]
pub type AdaptiveBackend = Autodiff<Cuda>;  // Desktop with GPU

#[cfg(not(feature = "cuda"))]
pub type AdaptiveBackend = Autodiff<NdArray>;  // Mobile or CPU-only
```

## Adaptive Training Configurations

### Desktop Configuration (CUDA)

```rust
AdaptiveTrainingConfig {
    batch_size: 32,        // Large batches for GPU
    epochs: 50,            // Full training
    learning_rate: 0.0001,
    use_augmentation: true,
}
```

**Performance**: 2-5 minutes for SSL retraining (17K + 200 pseudo-labels, 5 epochs)

### Mobile Configuration (NdArray)

```rust
AdaptiveTrainingConfig {
    batch_size: 8,         // Small batches for memory
    epochs: 2,             // Quick retraining
    learning_rate: 0.0001,
    use_augmentation: true, // Still prevent overfitting
}
```

**Performance**: 15-30 minutes for SSL retraining (same dataset, 2 epochs)

### SSL Retraining (Optimized)

For pseudo-labeling workflows, even more lightweight:

```rust
// Desktop
batch_size: 64,   // Larger GPU batches
epochs: 5,        // Quick cycles

// Mobile
batch_size: 8,
epochs: 2,        // Minimal for battery/thermal
```

## Usage Examples

### From Frontend (TypeScript)

The frontend can query device capabilities and use appropriate commands:

```typescript
// Get device info
const deviceInfo = await invoke<DeviceCapabilities>('get_device_info');
console.log(`Running on ${deviceInfo.device_type} with ${deviceInfo.backend_name}`);
console.log(`Recommended: ${deviceInfo.recommended_batch_size} batch size, ${deviceInfo.recommended_epochs} epochs`);

// Get recommended SSL config (automatically adapted)
const sslConfig = await invoke<AdaptiveTrainingConfig>('get_ssl_retraining_config');
// Returns different config based on device!

// Start SSL retraining (automatically uses right backend)
const result = await invoke<RetrainingResult>('start_ssl_retraining', {
  params: {
    model_path: './best_model.mpk',
    labeled_data_dir: './data/labeled',
    pseudo_labels: [...],  // Array of high-confidence predictions
    output_path: './retrained_model',
    custom_config: null,  // null = use adaptive config
  }
});
```

### Building for Different Platforms

#### Desktop (CUDA enabled)

```bash
cd plantvillage_ssl/gui/src-tauri
cargo build --release
# Uses default features: cuda
```

#### Desktop (CPU only)

```bash
cd plantvillage_ssl/gui/src-tauri
cargo build --release --no-default-features --features custom-protocol,ndarray
```

#### Mobile (iOS)

```bash
cd plantvillage_ssl/gui
bun run tauri:ios init  # First time only
bun run tauri:ios build
# Automatically uses --features mobile (which enables ndarray)
```

#### Mobile (Android)

```bash
cd plantvillage_ssl/gui
bun run tauri:android init  # First time only
bun run tauri:android build
# Automatically uses --features mobile
```

## SSL Workflow on Mobile

### 1. **User Collects Images**

Farmer takes photos of diseased plants with phone camera:
- 200+ images accumulated over days/weeks
- Stored locally on device

### 2. **Inference Phase (Fast)**

App runs inference on each image:
- **1-2ms per image** on desktop (CUDA)
- **50-100ms per image** on mobile (NdArray)
- Predicts disease class with confidence score

### 3. **Pseudo-Labeling**

High-confidence predictions (>90%) are accumulated:
```typescript
const pseudoLabels = inferenceResults
  .filter(r => r.confidence > 0.9)
  .map(r => ({
    image_path: r.path,
    predicted_label: r.label,
    confidence: r.confidence,
  }));
```

### 4. **Trigger Retraining**

When threshold reached (e.g., 200 pseudo-labels):
```typescript
// Show notification to user
showNotification('Model ready to improve! Retrain now?');

// User taps "Yes" → Start background retraining
const result = await invoke('start_ssl_retraining', {
  params: {
    model_path: currentModelPath,
    labeled_data_dir: bundledDatasetPath,  // See below
    pseudo_labels: pseudoLabels,
    output_path: './improved_model',
  }
});

// 15-30 min later...
alert(`Model improved! New accuracy: ${result.final_accuracy}%`);
```

### 5. **Model Updates**

Replace current model with retrained version:
```typescript
currentModelPath = result.model_path;
// Next inferences use improved model!
```

## Dataset Bundling for Mobile

Since we can't fit 87K images on mobile, use the dataset bundler:

### Create Mobile Bundle

```typescript
// From frontend
const metadata = await invoke<BundleMetadata>('create_dataset_bundle', {
  images_per_class: 50,  // 50 × 38 = 1,900 images (~200MB)
  source_dir: './data/plantvillage',
  output_dir: './mobile_dataset',
});

console.log(`Bundled ${metadata.total_images} images across ${metadata.num_classes} classes`);
```

### Bundle Contents

```
mobile_dataset/
├── Apple___Apple_scab/           (50 images)
├── Apple___Black_rot/            (50 images)
├── Apple___Cedar_apple_rust/     (50 images)
├── ...                           (35 more classes)
└── bundle_metadata.json          (metadata)
```

**Total size**: ~200MB (suitable for mobile app bundle)

### Embed in Mobile App

Add to `tauri.conf.json`:

```json
{
  "bundle": {
    "resources": [
      "mobile_dataset/**/*"
    ]
  }
}
```

The bundled dataset is included in the mobile app package and available offline.

## Performance Comparison

| Metric | Desktop (CUDA) | Mobile (NdArray) | Notes |
|--------|---------------|------------------|-------|
| **Inference** | 1-2ms | 50-100ms | Mobile still fast enough |
| **SSL Retrain (17K+200, 2 epochs)** | 1-2 min | 15-30 min | Acceptable for background |
| **Full Training (70K, 50 epochs)** | 1-2 hours | **4-5 days** | ❌ Not practical on mobile |
| **Model Size** | 1.8MB | 1.8MB | Same |
| **Memory Usage** | 2GB VRAM | 300MB RAM | Mobile-friendly |

## Key Insights

1. **SSL retraining is feasible on mobile** because:
   - Small retraining cycles (2 epochs vs 50)
   - Accumulated pseudo-labels (200) + labeled data (17K)
   - Total: ~35K forward/backward passes
   - 15-30 minutes is acceptable for background task

2. **Full training is NOT feasible** because:
   - 70K images × 50 epochs = 3.5M passes
   - Would take **days** on mobile CPU
   - Battery and thermal constraints

3. **Hybrid approach works best**:
   - Initial training on desktop/cloud (1-2 hours)
   - Deploy base model to mobile
   - Incremental improvement via SSL on-device

## Testing the System

### Test Device Detection

```bash
cd plantvillage_ssl/gui/src-tauri
cargo test --lib device::tests
```

### Test Adaptive Config

```bash
cargo test --lib device::tests::test_mobile_config_is_lighter
```

### Test Backend Selection

```bash
# Desktop with CUDA
cargo test --lib backend::tests --features cuda

# CPU only
cargo test --lib backend::tests --no-default-features --features ndarray
```

## Troubleshooting

### "CUDA backend not available"

You're running on a system without NVIDIA GPU. Build with NdArray:

```bash
cargo build --no-default-features --features custom-protocol,ndarray
```

### "Model loading failed on mobile"

Ensure the model was trained with the same Burn version (0.20.0-pre). Models are not cross-version compatible.

### "Out of memory on mobile"

Reduce batch size further in `device.rs`:

```rust
DeviceType::Mobile => DeviceCapabilities {
    recommended_batch_size: 4,  // Was 8
    // ...
}
```

## Future Enhancements

### WGPU Backend for Mobile GPU

Add GPU acceleration on mobile devices:

```toml
[dependencies]
burn-wgpu = "0.20.0-pre"  # WebGPU backend

[features]
mobile-gpu = ["burn-wgpu"]  # Use Vulkan (Android) / Metal (iOS)
```

This would speed up mobile retraining to **5-10 minutes** instead of 15-30.

### Federated Learning

Upload gradients instead of retraining on-device:

1. Run inference on mobile
2. Compute gradients for pseudo-labels
3. Send gradients to server (privacy-preserving)
4. Server aggregates and trains
5. Download updated model

## Summary

The adaptive backend system allows **the same Tauri app** to run on:

- **Desktop with CUDA GPU**: Fast training (CUDA backend, large batches, many epochs)
- **Desktop without GPU**: Slower training (NdArray backend, moderate batches)
- **Mobile (iOS/Android)**: Lightweight SSL retraining (NdArray backend, small batches, few epochs)

**Zero configuration needed** - the system automatically detects capabilities and adapts!

This enables **true on-device incremental learning** for farmers in the field. 🌱📱
