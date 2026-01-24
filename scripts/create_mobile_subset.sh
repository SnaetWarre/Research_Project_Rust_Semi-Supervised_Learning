#!/bin/bash
# Script to create a larger subset of PlantVillage dataset for mobile demo

set -e

FULL_DATASET_DIR="data/plantvillage_full"
SUBSET_DIR="data/farmer_demo_large"
TARGET_IMAGES_PER_CLASS=100  # Adjust as needed for mobile storage

echo "Creating larger subset for mobile demo..."

# Remove old subset if exists
rm -rf "$SUBSET_DIR"
mkdir -p "$SUBSET_DIR"

# Check if full dataset exists
if [ ! -d "$FULL_DATASET_DIR" ]; then
    echo "❌ Full dataset not found in $FULL_DATASET_DIR"
    echo "Please download PlantVillage dataset from Kaggle:"
    echo "https://www.kaggle.com/datasets/emmarex/plantdisease"
    echo "Extract it to $FULL_DATASET_DIR"
    exit 1
fi

echo "Found full dataset with $(find "$FULL_DATASET_DIR" -name "*.JPG" | wc -l) images"

# Create subset with balanced classes
for class_dir in "$FULL_DATASET_DIR"/*/; do
    class_name=$(basename "$class_dir")
    echo "Processing class: $class_name"

    # Create class directory in subset
    mkdir -p "$SUBSET_DIR/$class_name"

    # Get all images in class
    images=("$class_dir"/*.JPG)

    # Take up to TARGET_IMAGES_PER_CLASS randomly
    if [ ${#images[@]} -gt $TARGET_IMAGES_PER_CLASS ]; then
        # Shuffle and take first N
        printf '%s\n' "${images[@]}" | shuf | head -n $TARGET_IMAGES_PER_CLASS | xargs -I {} cp {} "$SUBSET_DIR/$class_name/"
        echo "  → Selected $TARGET_IMAGES_PER_CLASS images"
    else
        cp "${images[@]}" "$SUBSET_DIR/$class_name/"
        echo "  → Copied all ${#images[@]} images (less than target)"
    fi
done

# Count final subset
total_images=$(find "$SUBSET_DIR" -name "*.JPG" | wc -l)
echo ""
echo "✅ Subset created in $SUBSET_DIR"
echo "Total images: $total_images"
echo "Classes: $(ls "$SUBSET_DIR" | wc -l)"

# Update the data directory path in config
echo ""
echo "To use this subset, update the data_dir in your config to: $SUBSET_DIR"
echo "Or rename $SUBSET_DIR to data/farmer_demo"</content>
<parameter name="filePath">scripts/create_mobile_subset.sh