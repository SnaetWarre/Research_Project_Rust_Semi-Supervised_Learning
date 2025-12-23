#!/usr/bin/env python3
"""
PlantVillage Dataset Downloader

This script downloads and organizes the PlantVillage dataset for use with
the plantvillage_ssl Rust project. The dataset contains 61,486 images of
plant leaves categorized into 39 classes (38 disease classes + healthy).

Usage:
    python download_dataset.py [--output-dir OUTPUT_DIR] [--kaggle]

Requirements:
    pip install requests tqdm kaggle

Dataset Source:
    - Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    - Direct: Various mirrors available

Author: Warre Snaet
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Progress bars will be disabled.")
    tqdm = None


def download_from_kaggle(output_dir: Path) -> bool:
    """Download dataset from Kaggle using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        print("\nMake sure you have set up your Kaggle credentials:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New Token' to download kaggle.json")
        print(
            "3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)"
        )
        return False

    print("Downloading PlantVillage dataset from Kaggle...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        api.dataset_download_files(
            "abdallahalidev/plantvillage-dataset",
            path=str(output_dir),
            unzip=True,
        )
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def organize_dataset(raw_dir: Path, output_dir: Path) -> bool:
    """Organize the downloaded dataset into a clean structure."""
    print("\nOrganizing dataset...")

    # Find the image directories
    # The dataset might be in plantvillage-dataset/color or similar structure
    possible_paths = [
        raw_dir / "plantvillage dataset" / "color",
        raw_dir / "PlantVillage" / "color",
        raw_dir / "color",
        raw_dir / "plantvillage dataset" / "segmented",
        raw_dir / "segmented",
    ]

    source_dir = None
    for path in possible_paths:
        if path.exists():
            source_dir = path
            print(f"Found image directory: {source_dir}")
            break

    if source_dir is None:
        print("Warning: Could not find standard PlantVillage directory structure.")
        print("Looking for any directory with plant class folders...")

        # Look for directories that might contain class folders
        for item in raw_dir.rglob("*"):
            if item.is_dir() and "Apple" in item.name:
                source_dir = item.parent
                print(f"Found class directories in: {source_dir}")
                break

    if source_dir is None:
        print("Error: Could not locate image directories.")
        print(f"Please check the contents of: {raw_dir}")
        return False

    # Count classes and images
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(class_dirs)} classes:")

    total_images = 0
    for class_dir in sorted(class_dirs):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        images += list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
        print(f"  - {class_dir.name}: {len(images)} images")
        total_images += len(images)

    print(f"\nTotal: {total_images} images across {len(class_dirs)} classes")

    # Create organized output structure
    organized_dir = output_dir / "organized"
    organized_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying to organized structure: {organized_dir}")

    for class_dir in tqdm(sorted(class_dirs)) if tqdm else sorted(class_dirs):
        dest_class_dir = organized_dir / class_dir.name
        dest_class_dir.mkdir(exist_ok=True)

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                shutil.copy2(img_file, dest_class_dir / img_file.name)

    print("\nDataset organized successfully!")
    print(f"Use this path in your Rust code: {organized_dir}")

    return True


def verify_dataset(data_dir: Path) -> bool:
    """Verify the dataset is correctly organized."""
    print("\nVerifying dataset...")

    if not data_dir.exists():
        print(f"Error: Directory does not exist: {data_dir}")
        return False

    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if len(class_dirs) == 0:
        print("Error: No class directories found!")
        return False

    # Expected classes (subset for verification)
    expected_classes = [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Tomato___Late_blight",
        "Tomato___healthy",
    ]

    found_expected = sum(1 for c in expected_classes if (data_dir / c).exists())

    if found_expected < len(expected_classes):
        print(
            f"Warning: Only found {found_expected}/{len(expected_classes)} expected classes"
        )
        print("The dataset structure might be different than expected.")

    # Count total images
    total_images = 0
    min_images = float("inf")
    max_images = 0

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.*"))
        image_count = len(
            [f for f in images if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        total_images += image_count
        min_images = min(min_images, image_count)
        max_images = max(max_images, image_count)

    print(f"\n✅ Dataset verification complete:")
    print(f"   - Classes: {len(class_dirs)}")
    print(f"   - Total images: {total_images}")
    print(f"   - Min images per class: {min_images}")
    print(f"   - Max images per class: {max_images}")

    if total_images < 1000:
        print(
            "\n⚠️ Warning: Dataset seems smaller than expected (should have ~61,000 images)"
        )
        return False

    return True


def create_splits_config(data_dir: Path) -> None:
    """Create a default splits configuration file."""
    config = {
        "test_fraction": 0.10,
        "validation_fraction": 0.10,
        "labeled_fraction": 0.20,
        "stream_fraction": 0.60,
        "seed": 42,
        "stratified": True,
    }

    import json

    config_path = data_dir.parent / "split_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nCreated split configuration: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize the PlantVillage dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plantvillage",
        help="Output directory for the dataset (default: data/plantvillage)",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Download from Kaggle (requires Kaggle API credentials)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an existing dataset, don't download",
    )
    parser.add_argument(
        "--organize",
        type=str,
        help="Organize an already-downloaded dataset from this path",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("PlantVillage Dataset Downloader")
    print("=" * 60)

    if args.verify_only:
        organized_dir = output_dir / "organized"
        if organized_dir.exists():
            verify_dataset(organized_dir)
        else:
            verify_dataset(output_dir)
        return

    if args.organize:
        raw_dir = Path(args.organize)
        if organize_dataset(raw_dir, output_dir):
            verify_dataset(output_dir / "organized")
            create_splits_config(output_dir / "organized")
        return

    if args.kaggle:
        if download_from_kaggle(output_dir):
            if organize_dataset(output_dir, output_dir):
                verify_dataset(output_dir / "organized")
                create_splits_config(output_dir / "organized")
    else:
        print("\nNo download method specified.")
        print("\nOptions:")
        print("1. Download from Kaggle (requires API credentials):")
        print(f"   python {sys.argv[0]} --kaggle")
        print("\n2. Download manually and organize:")
        print(
            "   a. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
        )
        print("   b. Download and extract the dataset")
        print(f"   c. Run: python {sys.argv[0]} --organize /path/to/extracted/dataset")
        print("\n3. Verify an existing dataset:")
        print(f"   python {sys.argv[0]} --verify-only --output-dir /path/to/dataset")


if __name__ == "__main__":
    main()
