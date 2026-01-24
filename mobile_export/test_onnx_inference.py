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
from load_weights import CLASS_NAMES, IMAGE_SIZE, PlantClassifier, load_weights_from_json


def preprocess_image(image_path: Path, method="bilinear") -> np.ndarray:
    """
    Preprocess image exactly as PWA does (or should do).
    
    Returns: numpy array of shape (1, 3, 128, 128) ready for inference
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize to 128x128 (same as model expects)
    if method == "lanczos":
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    else:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)

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
        import onnxruntime as ort  # type: ignore[import-not-found]
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


def evaluate_folder(
    folder_path: Path,
    weights_path: Path,
    onnx_path: Path,
    method: str,
    max_images: int | None,
) -> dict:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        print("\n⚠️  onnxruntime not installed. Skipping ONNX test.")
        print("   Install with: pip install onnxruntime")
        sys.exit(1)

    model = PlantClassifier(num_classes=38)
    load_weights_from_json(model, weights_path)
    model.eval()

    session = ort.InferenceSession(str(onnx_path))

    image_paths = sorted(
        [p for p in folder_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        print(f"⚠️  No images found in {folder_path}")
        return {}

    expected_name = folder_path.name
    expected_idx = CLASS_NAMES.index(expected_name) if expected_name in CLASS_NAMES else None

    results = {
        "folder": str(folder_path),
        "total": 0,
        "correct": 0,
        "expected_idx": expected_idx,
        "top_mismatch": {},
    }

    for image_path in image_paths:
        img = preprocess_image(image_path, method=method)
        img_tensor = torch.from_numpy(img).float()

        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].numpy()
        top_idx = int(np.argmax(probs))

        results["total"] += 1
        if expected_idx is not None and top_idx == expected_idx:
            results["correct"] += 1
        else:
            results["top_mismatch"][top_idx] = results["top_mismatch"].get(top_idx, 0) + 1

        outputs = session.run(None, {"image": img.astype(np.float32)})
        onnx_logits = outputs[0][0]
        exp = np.exp(onnx_logits - np.max(onnx_logits))
        onnx_probs = exp / exp.sum()
        onnx_top = int(np.argmax(onnx_probs))

        if onnx_top != top_idx:
            results["top_mismatch"][f"onnx_vs_torch_{onnx_top}"] = (
                results["top_mismatch"].get(f"onnx_vs_torch_{onnx_top}", 0) + 1
            )

    return results


def print_folder_results(results: dict) -> None:
    if not results:
        return

    total = results["total"]
    correct = results["correct"]
    expected_idx = results["expected_idx"]
    expected_name = CLASS_NAMES[expected_idx] if expected_idx is not None else "Unknown"
    accuracy = (correct / total * 100.0) if total else 0.0

    print("\n" + "=" * 60)
    print(f"Folder: {results['folder']}")
    print(f"Expected class: {expected_name} (idx {expected_idx})")
    print(f"Total images: {total}")
    print(f"Correct top-1: {correct} ({accuracy:.2f}%)")

    if results["top_mismatch"]:
        print("Top mismatches:")
        for key, count in sorted(results["top_mismatch"].items(), key=lambda x: x[1], reverse=True)[:5]:
            if isinstance(key, int):
                name = CLASS_NAMES[key] if key < len(CLASS_NAMES) else f"Unknown_{key}"
                print(f"  {name}: {count}")
            else:
                print(f"  {key}: {count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to test image",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Path to a folder to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Evaluate every class folder under this directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit images per folder",
    )
    parser.add_argument(
        "--preprocess",
        choices=["bilinear", "lanczos"],
        default="bilinear",
        help="Resize method",
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

    if not args.weights.exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    if not args.onnx.exists():
        print(f"❌ ONNX model not found: {args.onnx}")
        sys.exit(1)

    if args.image:
        if not args.image.exists():
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)

        print(f"Testing with image: {args.image}")
        print(f"Expected class: {args.image.parent.name}\n")

        pytorch_probs, _ = test_pytorch_model(args.image, args.weights)
        onnx_probs, _ = test_onnx_model(args.image, args.onnx)
        compare_results(pytorch_probs, onnx_probs)
        print("\n" + "=" * 60)
        print("Test Complete")
        print("=" * 60)
        return

    if args.folder:
        if not args.folder.exists():
            print(f"❌ Folder not found: {args.folder}")
            sys.exit(1)

        results = evaluate_folder(args.folder, args.weights, args.onnx, args.preprocess, args.max_images)
        print_folder_results(results)
        return

    if args.data_dir:
        if not args.data_dir.exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            sys.exit(1)

        class_folders = [p for p in args.data_dir.iterdir() if p.is_dir()]
        if not class_folders:
            print(f"❌ No class folders found in {args.data_dir}")
            sys.exit(1)

        overall_total = 0
        overall_correct = 0
        mismatch_totals = {}

        for folder in sorted(class_folders):
            results = evaluate_folder(folder, args.weights, args.onnx, args.preprocess, args.max_images)
            print_folder_results(results)

            overall_total += results.get("total", 0)
            overall_correct += results.get("correct", 0)
            for key, count in results.get("top_mismatch", {}).items():
                mismatch_totals[key] = mismatch_totals.get(key, 0) + count

        accuracy = (overall_correct / overall_total * 100.0) if overall_total else 0.0
        print("\n" + "=" * 60)
        print("Overall")
        print("=" * 60)
        print(f"Total images: {overall_total}")
        print(f"Correct top-1: {overall_correct} ({accuracy:.2f}%)")

        if mismatch_totals:
            print("Top overall mismatches:")
            for key, count in sorted(mismatch_totals.items(), key=lambda x: x[1], reverse=True)[:10]:
                if isinstance(key, int):
                    name = CLASS_NAMES[key] if key < len(CLASS_NAMES) else f"Unknown_{key}"
                    print(f"  {name}: {count}")
                else:
                    print(f"  {key}: {count}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
