#!/bin/bash
set -e

# Experiment 1: Data Augmentation Impact
echo "========================================================"
echo "RUNNING EXPERIMENT: Data Augmentation Impact"
echo "========================================================"

# 1. Train WITHOUT Augmentation (Baseline)
echo "1. Training Baseline (No Augmentation)..."
./plantvillage_ssl/target/release/plantvillage_ssl train \
    --epochs 1 \
    --cuda \
    --labeled-ratio 0.2 \
    --output-dir "output/experiments/no_aug" \
    --data-dir "plantvillage_ssl/data/plantvillage" \
    > output/experiments/log_no_aug.txt 2>&1

# 2. Train WITH Augmentation
echo "2. Training With Augmentation..."
./plantvillage_ssl/target/release/plantvillage_ssl train \
    --epochs 1 \
    --cuda \
    --labeled-ratio 0.2 \
    --augmentation \
    --output-dir "output/experiments/with_aug" \
    --data-dir "plantvillage_ssl/data/plantvillage" \
    > output/experiments/log_with_aug.txt 2>&1

# Experiment 2: Pseudo-labeling Effectiveness (SSL)
echo "========================================================"
echo "RUNNING EXPERIMENT: SSL Effectiveness"
echo "========================================================"

# We use the model trained with augmentation as the starting point
# and simulate 1 'day' of streaming data to see pseudo-label generation stats
./plantvillage_ssl/target/release/plantvillage_ssl simulate \
    --model "output/experiments/with_aug/best_model.mpk" \
    --data-dir "plantvillage_ssl/data/plantvillage" \
    --days 1 \
    --images-per-day 200 \
    --confidence-threshold 0.85 \
    --output-dir "output/experiments/ssl_sim" \
    --labeled-ratio 0.2 \
    > output/experiments/log_ssl.txt 2>&1

echo "Experiments complete. Parsing results..."

# Extract Metrics
echo "--- RESULTS ---"
echo "No Augmentation Final Accuracy:"
grep "Accuracy" output/experiments/log_no_aug.txt | tail -n 1
echo "With Augmentation Final Accuracy:"
grep "Accuracy" output/experiments/log_with_aug.txt | tail -n 1
echo "SSL Pseudo-labels Generated:"
grep "Generated" output/experiments/log_ssl.txt || echo "Check log_ssl.txt"
