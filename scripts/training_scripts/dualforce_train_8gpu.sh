#!/bin/bash
# ============================================================
# DualForce Training Launch Script (8x A100 80GB)
# ============================================================
# Usage:
#   bash scripts/training_scripts/dualforce_train_8gpu.sh
#
# To override config:
#   bash scripts/training_scripts/dualforce_train_8gpu.sh \
#       --cfg-options trainer.max_steps=50000 data.batch_size=1
# ============================================================

set -e

# Environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# Paths
CONFIG="configs/dualforce/dualforce_train_8gpu.py"
ACCELERATE_CONFIG="configs/dualforce/accelerate/fsdp_8gpu.yaml"

echo "=========================================="
echo " DualForce Training (8x GPU, FSDP)"
echo "=========================================="
echo " Config:     ${CONFIG}"
echo " Accelerate: ${ACCELERATE_CONFIG}"
echo "=========================================="

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    scripts/training_scripts/dualforce_train.py ${CONFIG} "$@"
