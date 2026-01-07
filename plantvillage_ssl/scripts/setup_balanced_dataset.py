#!/usr/bin/env python3
"""
PlantVillage Dataset Setup Script

This script creates a balanced, clean dataset from the PlantVillage raw data.
It ensures equal representation of all classes to prevent model bias.

Usage:
    python setup_balanced_dataset.py [--samples-per-class N] [--seed S]
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


# Standard PlantVillage class names (38 classes)
EXPECTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def normalize_class_name(name: str) -> str:
    """Normalize class name to handle variations in naming."""
    # Handle common variations
    name = name.replace("_", "_").strip()
    
    # Fix specific known issues
    replacements = {
        "Cercospora_leaf_spot_Gray_leaf_spot": "Cercospora_leaf_spot Gray_leaf_spot",
        "Spider_mites_Two-spotted_spider_mite": "Spider_mites Two-spotted_spider_mite",
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
    
    return name


def find_source_images(raw_dir: Path) -> dict:
    """
    Find all source images, preferring color over grayscale/segmented.
    Returns a dict mapping class_name -> list of image paths.
    """
    class_images = defaultdict(list)
    
    # Check different possible structures
    color_dir = raw_dir / "plantvillage dataset" / "color"
    
    if color_dir.exists():
        # Kaggle format: raw/plantvillage dataset/color/{class}/*.jpg
        source_dir = color_dir
    else:
        # Direct format: raw/{class}/*.jpg or organized/{class}/*.jpg
        source_dir = raw_dir
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return class_images
    
    print(f"Scanning source directory: {source_dir}")
    
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = normalize_class_name(class_dir.name)
        
        # Find all images in this class
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(class_dir.glob(ext))
        
        if images:
            class_images[class_name] = sorted(images)
    
    return class_images


def create_balanced_dataset(
    source_images: dict,
    output_dir: Path,
    samples_per_class: int,
    seed: int = 42,
    allow_oversampling: bool = False
) -> dict:
    """
    Create a balanced dataset with equal samples per class.
    
    If a class has fewer samples than requested:
      - Without oversampling: all samples are used (unbalanced)
      - With oversampling: images are duplicated to reach target count
    
    If a class has more samples, random sampling is performed.
    """
    random.seed(seed)
    
    # Clean output directory
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True)
    
    stats = {}
    
    mode = "with oversampling" if allow_oversampling else "without oversampling"
    print(f"\nCreating balanced dataset ({mode}) with {samples_per_class} samples per class...")
    print("=" * 70)
    
    for class_name in sorted(source_images.keys()):
        images = source_images[class_name]
        available = len(images)
        
        # Create class directory
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True)
        
        if available >= samples_per_class:
            # Downsample: randomly select
            selected = random.sample(images, samples_per_class)
            used = samples_per_class
            oversampled = 0
        elif allow_oversampling:
            # Oversample: use all images and duplicate to reach target
            selected = images.copy()
            oversampled = samples_per_class - available
            
            # Randomly duplicate images to reach target
            while len(selected) < samples_per_class:
                selected.append(random.choice(images))
            
            random.shuffle(selected)
            used = samples_per_class
            print(f"  {class_name}: {available} images + {oversampled} oversampled = {used}")
        else:
            # No oversampling: use what we have
            selected = images
            used = available
            oversampled = 0
            print(f"  WARNING: {class_name} only has {available} images (< {samples_per_class})")
        
        # Copy images
        for i, src_path in enumerate(selected):
            # Use consistent naming
            dst_path = class_output_dir / f"{class_name}_{i:04d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, dst_path)
        
        stats[class_name] = {
            "available": available, 
            "used": used,
            "oversampled": oversampled if allow_oversampling else 0
        }
        
        if not allow_oversampling or available >= samples_per_class:
            print(f"  {class_name}: {used}/{available} images")
    
    return stats


def print_summary(stats: dict, output_dir: Path):
    """Print a summary of the created dataset."""
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    total_available = sum(s["available"] for s in stats.values())
    total_used = sum(s["used"] for s in stats.values())
    num_classes = len(stats)
    
    # Find min/max
    min_class = min(stats.items(), key=lambda x: x[1]["available"])
    max_class = max(stats.items(), key=lambda x: x[1]["available"])
    
    print(f"\nTotal classes: {num_classes}")
    print(f"Total images available: {total_available:,}")
    print(f"Total images used: {total_used:,}")
    print(f"\nOriginal imbalance ratio: {max_class[1]['available']/min_class[1]['available']:.1f}:1")
    print(f"  Largest class: {max_class[0]} ({max_class[1]['available']} images)")
    print(f"  Smallest class: {min_class[0]} ({min_class[1]['available']} images)")
    
    # Check balance in output
    min_used = min(s["used"] for s in stats.values())
    max_used = max(s["used"] for s in stats.values())
    
    total_oversampled = sum(s.get("oversampled", 0) for s in stats.values())
    
    if min_used == max_used:
        print(f"\nNew dataset is PERFECTLY BALANCED: {min_used} images per class")
        if total_oversampled > 0:
            print(f"  (includes {total_oversampled:,} oversampled duplicates)")
    else:
        print(f"\nNew dataset imbalance ratio: {max_used/min_used:.1f}:1")
        print(f"  (Some classes had fewer samples than requested)")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Dataset size: {sum(1 for _ in output_dir.rglob('*.jpg')) + sum(1 for _ in output_dir.rglob('*.png')):,} images")


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced PlantVillage dataset"
    )
    parser.add_argument(
        "--samples-per-class", "-n",
        type=int,
        default=500,
        help="Number of samples per class (default: 500, use smallest class size if smaller)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data directory (default: auto-detect)"
    )
    parser.add_argument(
        "--use-min-class-size",
        action="store_true",
        help="Use the minimum class size instead of --samples-per-class"
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Allow oversampling (duplicating) images for smaller classes to reach target"
    )
    args = parser.parse_args()
    
    # Find data directory
    script_dir = Path(__file__).parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / "data" / "plantvillage"
    
    raw_dir = data_dir / "raw"
    organized_dir = data_dir / "organized"
    output_dir = data_dir / "balanced"
    
    print("PlantVillage Dataset Balancer")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find source images
    if raw_dir.exists():
        source_images = find_source_images(raw_dir)
    elif organized_dir.exists():
        source_images = find_source_images(organized_dir)
    else:
        print(f"ERROR: No source data found in {data_dir}")
        print("Please download the PlantVillage dataset first.")
        return 1
    
    if not source_images:
        print("ERROR: No images found in source directory")
        return 1
    
    print(f"\nFound {len(source_images)} classes")
    
    # Determine samples per class
    min_available = min(len(imgs) for imgs in source_images.values())
    
    if args.use_min_class_size:
        samples_per_class = min_available
        print(f"Using minimum class size: {samples_per_class}")
    elif args.oversample:
        samples_per_class = args.samples_per_class
        print(f"Using {samples_per_class} samples per class with oversampling enabled")
    else:
        samples_per_class = min(args.samples_per_class, min_available)
        if samples_per_class < args.samples_per_class:
            print(f"Note: Smallest class has {min_available} images, capping at that")
            print(f"      Use --oversample to reach {args.samples_per_class} per class")
    
    # Create balanced dataset
    stats = create_balanced_dataset(
        source_images,
        output_dir,
        samples_per_class,
        args.seed,
        allow_oversampling=args.oversample
    )
    
    # Print summary
    print_summary(stats, output_dir)
    
    # Create a config file
    config_path = output_dir / "dataset_config.json"
    import json
    config = {
        "samples_per_class": samples_per_class,
        "seed": args.seed,
        "num_classes": len(stats),
        "total_images": sum(s["used"] for s in stats.values()),
        "balanced": True,
        "class_stats": stats,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig saved to: {config_path}")
    print("\nDone! Use 'data/plantvillage/balanced' as your dataset path.")
    
    return 0


if __name__ == "__main__":
    exit(main())
