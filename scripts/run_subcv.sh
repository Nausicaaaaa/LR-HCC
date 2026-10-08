#!/bin/bash
# 5-fold sub-CV training pipeline
set -e
cd /mnt/data/KASR/Dengsiyi/LR-HCC

eval "$(conda shell.bash hook)"
conda activate nnunetv2

echo "=== Starting 5-fold sub-CV training ==="
echo "$(date)"

# Step 1: Train + predict
python shap_outputs/train_5fold_subcv.py --mode train_predict --epochs 50 --lr 5e-5

# Step 2: Compute accuracy
python shap_outputs/train_5fold_subcv.py --mode accuracy

echo "=== Done ==="
echo "$(date)"
