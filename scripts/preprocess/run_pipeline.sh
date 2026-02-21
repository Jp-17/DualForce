#!/bin/bash
# DualForce Data Preprocessing Pipeline
# Run all preprocessing steps in sequence.
#
# Usage:
#   bash scripts/preprocess/run_pipeline.sh \
#       --input_dir /path/to/raw_videos \
#       --output_dir /path/to/processed_data \
#       --vae_path /path/to/MOVA-360p/video_vae \
#       --device cuda

set -e

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
VAE_PATH=""
HUBERT_MODEL="facebook/hubert-large-ls960-ft"
DEVICE="cuda"
TARGET_FPS=25
TARGET_SIZE=512

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --vae_path) VAE_PATH="$2"; shift 2 ;;
        --hubert_model) HUBERT_MODEL="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --target_fps) TARGET_FPS="$2"; shift 2 ;;
        --target_size) TARGET_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --input_dir <path> --output_dir <path> [--vae_path <path>] [--device cuda]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo " DualForce Preprocessing Pipeline"
echo "=========================================="
echo " Input:  $INPUT_DIR"
echo " Output: $OUTPUT_DIR"
echo " Device: $DEVICE"
echo "=========================================="

# Step 1: Face Detection & Cropping
echo ""
echo "[Step 1/7] Face detection, cropping, FPS normalization..."
python "$SCRIPT_DIR/01_face_detect_crop.py" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --target_fps "$TARGET_FPS" \
    --target_size "$TARGET_SIZE"

# Step 2: Quality Filtering
echo ""
echo "[Step 2/7] Quality filtering..."
python "$SCRIPT_DIR/02_quality_filter.py" \
    --data_dir "$OUTPUT_DIR"

# Step 3: Video Latent Extraction (requires GPU)
if [ -n "$VAE_PATH" ]; then
    echo ""
    echo "[Step 3/7] Extracting video latents..."
    python "$SCRIPT_DIR/03_extract_video_latents.py" \
        --data_dir "$OUTPUT_DIR" \
        --vae_path "$VAE_PATH" \
        --clip_list "$OUTPUT_DIR/filtered_clips.json" \
        --device "$DEVICE"
else
    echo ""
    echo "[Step 3/7] SKIPPED (no --vae_path provided)"
fi

# Step 4: Struct Latent Extraction
echo ""
echo "[Step 4/7] Extracting struct latents..."
python "$SCRIPT_DIR/04_extract_struct_latents.py" \
    --data_dir "$OUTPUT_DIR" \
    --clip_list "$OUTPUT_DIR/filtered_clips.json" \
    --device "$DEVICE" \
    --fallback

# Step 5: FLAME Parameter Extraction
echo ""
echo "[Step 5/7] Extracting FLAME parameters..."
python "$SCRIPT_DIR/05_extract_flame_params.py" \
    --data_dir "$OUTPUT_DIR" \
    --clip_list "$OUTPUT_DIR/filtered_clips.json" \
    --device "$DEVICE" \
    --fallback

# Step 6: Audio Feature Extraction
echo ""
echo "[Step 6/7] Extracting audio features & first frames..."
python "$SCRIPT_DIR/06_extract_audio_features.py" \
    --data_dir "$OUTPUT_DIR" \
    --hubert_model "$HUBERT_MODEL" \
    --clip_list "$OUTPUT_DIR/filtered_clips.json" \
    --device "$DEVICE"

# Step 7: Build Metadata
echo ""
echo "[Step 7/7] Building metadata.json..."
python "$SCRIPT_DIR/07_build_metadata.py" \
    --data_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Metadata: $OUTPUT_DIR/metadata.json"
echo "=========================================="
