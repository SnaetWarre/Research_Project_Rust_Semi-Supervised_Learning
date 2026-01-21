#!/bin/bash
# Script to extract ~50 images for farmer demo feature
# These images will be excluded from the regular SSL stream and used
# to simulate a farmer uploading images from their field

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
PLANTVILLAGE_DIR="$DATA_DIR/plantvillage"
FARMER_DEMO_DIR="$DATA_DIR/farmer_demo"
EXCLUSION_FILE="$DATA_DIR/farmer_demo_exclusions.txt"

# Number of images to extract (roughly balanced across classes)
# Increased to 500 to ensure reaching retrain threshold of 200 pseudo-labels
# (with ~90% acceptance rate at 0.9 confidence threshold)
TOTAL_IMAGES=500

echo "Setting up farmer demo images..."
echo "Project dir: $PROJECT_DIR"
echo "Data dir: $DATA_DIR"

# Check if plantvillage data exists
if [ ! -d "$PLANTVILLAGE_DIR/train" ]; then
    echo "Error: PlantVillage dataset not found at $PLANTVILLAGE_DIR"
    echo "Please run download_dataset.sh first."
    exit 1
fi

# Create farmer demo directory
mkdir -p "$FARMER_DEMO_DIR"

# Clear existing files if any
rm -rf "$FARMER_DEMO_DIR"/*
rm -f "$EXCLUSION_FILE"

# Initialize exclusion file
echo "# Farmer demo image exclusions - these paths are excluded from SSL stream" > "$EXCLUSION_FILE"
echo "# Generated on $(date)" >> "$EXCLUSION_FILE"
echo "" >> "$EXCLUSION_FILE"

# Track total copied
TOTAL_COPIED=0

# Get all classes
mapfile -t CLASSES < <(ls -1 "$PLANTVILLAGE_DIR/train")
NUM_CLASSES=${#CLASSES[@]}
echo "Found $NUM_CLASSES classes"

# Calculate images per class
IMAGES_PER_CLASS=$((TOTAL_IMAGES / NUM_CLASSES + 1))
echo "Extracting approximately $IMAGES_PER_CLASS images per class..."

# Process each class
for CLASS in "${CLASSES[@]}"; do
    if [ $TOTAL_COPIED -ge $TOTAL_IMAGES ]; then
        break
    fi
    
    CLASS_DIR="$PLANTVILLAGE_DIR/train/$CLASS"
    FARMER_CLASS_DIR="$FARMER_DEMO_DIR/$CLASS"
    
    if [ ! -d "$CLASS_DIR" ]; then
        continue
    fi
    
    # Create class subdirectory in farmer_demo
    mkdir -p "$FARMER_CLASS_DIR"
    
    # Calculate how many to copy from this class
    REMAINING=$((TOTAL_IMAGES - TOTAL_COPIED))
    TO_COPY=$((IMAGES_PER_CLASS < REMAINING ? IMAGES_PER_CLASS : REMAINING))
    
    # Get random images - use a temp file to handle spaces properly
    COPIED_FROM_CLASS=0
    
    # List all images, shuffle, and take first TO_COPY
    find "$CLASS_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | shuf | head -n "$TO_COPY" | while IFS= read -r IMG_PATH; do
        if [ $TOTAL_COPIED -ge $TOTAL_IMAGES ]; then
            break
        fi
        
        IMG_NAME=$(basename "$IMG_PATH")
        DST="$FARMER_CLASS_DIR/$IMG_NAME"
        
        # Copy the image
        cp "$IMG_PATH" "$DST"
        
        # Add to exclusion list (relative path from plantvillage dir)
        echo "train/$CLASS/$IMG_NAME" >> "$EXCLUSION_FILE"
        
        echo "$((TOTAL_COPIED + COPIED_FROM_CLASS + 1))" > /tmp/farmer_demo_count
        ((COPIED_FROM_CLASS++)) || true
    done
    
    # Update total from file (subshell doesn't update parent vars)
    if [ -f /tmp/farmer_demo_count ]; then
        TOTAL_COPIED=$(cat /tmp/farmer_demo_count)
    fi
    
done

rm -f /tmp/farmer_demo_count

# Count actual images copied
ACTUAL_COUNT=$(find "$FARMER_DEMO_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

echo ""
echo "Done! Extracted $ACTUAL_COUNT images to $FARMER_DEMO_DIR"
echo "Exclusion list saved to $EXCLUSION_FILE"
echo ""
echo "Folder structure:"
find "$FARMER_DEMO_DIR" -type d | head -10
echo ""
echo "Total images per class:"
for dir in "$FARMER_DEMO_DIR"/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f 2>/dev/null | wc -l)
        name=$(basename "$dir")
        if [ "$count" -gt 0 ]; then
            echo "  $name: $count"
        fi
    fi
done | head -15
echo "..."
echo ""
EXCL_LINES=$(grep -c "^train/" "$EXCLUSION_FILE" 2>/dev/null || echo "0")
echo "Exclusion file has $EXCL_LINES image entries"
