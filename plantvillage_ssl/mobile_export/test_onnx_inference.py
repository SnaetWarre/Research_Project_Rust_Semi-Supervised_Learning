#!/usr/bin/env python3
"""
Test ONNX model inference and compare with PyTorch reference.

This script tests that the ONNX model produces the same outputs as PyTorch
when given the same preprocessed input.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add mobile_export to path
sys.path.insert(0, str(Path(__file__).parent))
from load_weights import PlantClassifier, load_weights_from_json

# Class names (alphabetical order as in training)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___healthy", "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot", "Grape___Esca", "Grape___healthy", "Grape___Leaf_blight",
    "Orange___Citrus_greening", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight", "Potato___healthy",
    "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot", "Tomato___mosaic_virus", "Tomato___Yellow_Leaf_Curl_Virus"
]


def preprocess_image(image_path: Path, method="bilinear") -> np.ndarray:
    """
    Preprocess image exactly as PWA does (or should do).
    
    Returns: numpy array of shape (1, 3, 128, 128) ready for inference
    """
    # Load image
    img = Image.open(image_path)
    
    # Resize to 128x128 (same as model expects)
    if method == "bilinear":
        img = img.resize((128, 128), Image.BILINEAR)
    elif method == "lanczos":
        img = img.resize((128, 128), Image.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array [H, W, C]
    img = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # Convert to CHW format
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    
    # Add batch dimension
    img = np.expand_dims(img, 0)
    
    return img


def test_pytorch_model(image_path: Path, weights_path: Path):
    """Test PyTorch model with Burn weights."""
    print("=" * 60)
    print("Testing PyTorch Model (Reference)")
    print("=" * 60)
    
    # Load model
    model = PlantClassifier(num_classes=38)
    load_weights_from_json(model, weights_path)
    model.eval()
    
    # Preprocess image
    img = preprocess_image(image_path)
    img_tensor = torch.from_numpy(img).float()
    
    print(f"Input shape: {img_tensor.shape}")
    print(f"Input range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"Input mean: {img_tensor.mean():.3f}")
    
    # Per-channel stats
    for c, name in enumerate(['R', 'G', 'B']):
        channel = img_tensor[0, c]
        print(f"  {name} channel: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
    
    # Run inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].numpy()
    
    # Top 5
    top5_idx = np.argsort(probs)[::-1][:5]
    
    print(f"\nTop 5 predictions:")
    for i, idx in enumerate(top5_idx):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown_{idx}"
        print(f"  {i+1}. [{idx:2d}] {name:45s} {probs[idx]*100:5.2f}%")
    
    return probs, top5_idx[0]


def test_onnx_model(image_path: Path, onnx_path: Path):
    """Test ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n⚠️  onnxruntime not installed. Skipping ONNX test.")
        print("   Install with: pip install onnxruntime")
        return None, None
    
    print("\n" + "=" * 60)
    print("Testing ONNX Model")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession(str(onnx_path))
    
    # Preprocess image
    img = preprocess_image(image_path)
    
    print(f"Input shape: {img.shape}")
    print(f"Input range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Input mean: {img.mean():.3f}")
    
    # Per-channel stats
    for c, name in enumerate(['R', 'G', 'B']):
        channel = img[0, c]
        print(f"  {name} channel: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
    
    # Run inference
    outputs = session.run(None, {"image": img.astype(np.float32)})
    logits = outputs[0][0]
    
    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    
    # Top 5
    top5_idx = np.argsort(probs)[::-1][:5]
    
    print(f"\nTop 5 predictions:")
    for i, idx in enumerate(top5_idx):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown_{idx}"
        print(f"  {i+1}. [{idx:2d}] {name:45s} {probs[idx]*100:5.2f}%")
    
    return probs, top5_idx[0]


def compare_results(pytorch_probs, onnx_probs):
    """Compare PyTorch and ONNX results."""
    if onnx_probs is None:
        return
    
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    
    # Max absolute difference
    max_diff = np.abs(pytorch_probs - onnx_probs).max()
    mean_diff = np.abs(pytorch_probs - onnx_probs).mean()
    
    print(f"Max probability difference: {max_diff:.6f}")
    print(f"Mean probability difference: {mean_diff:.6f}")
    
    # Check if top predictions match
    pytorch_top = np.argmax(pytorch_probs)
    onnx_top = np.argmax(onnx_probs)
    
    if pytorch_top == onnx_top:
        print(f"✅ Top predictions match: class {pytorch_top}")
    else:
        print(f"❌ Top predictions differ!")
        print(f"   PyTorch: class {pytorch_top} ({CLASS_NAMES[pytorch_top]})")
        print(f"   ONNX:    class {onnx_top} ({CLASS_NAMES[onnx_top]})")
    
    # Tolerance check
    if max_diff < 0.01:
        print("✅ Results are nearly identical (max diff < 1%)")
    elif max_diff < 0.05:
        print("⚠️  Small differences detected (max diff < 5%)")
    else:
        print("❌ Significant differences detected (max diff >= 5%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("../data/plantvillage/balanced/Peach___healthy/Peach___healthy_0000.jpg"),
        help="Path to test image",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/weights.json"),
        help="Path to Burn weights JSON",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("pwa/model.onnx"),
        help="Path to ONNX model",
    )
    args = parser.parse_args()
    
    if not args.image.exists():
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    if not args.weights.exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)
    
    if not args.onnx.exists():
        print(f"❌ ONNX model not found: {args.onnx}")
        sys.exit(1)
    
    print(f"Testing with image: {args.image}")
    print(f"Expected class: {args.image.parent.name}\n")
    
    # Test PyTorch
    pytorch_probs, pytorch_top = test_pytorch_model(args.image, args.weights)
    
    # Test ONNX
    onnx_probs, onnx_top = test_onnx_model(args.image, args.onnx)
    
    # Compare
    compare_results(pytorch_probs, onnx_probs)
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
