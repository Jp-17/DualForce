#!/bin/bash
# ============================================================
# DualForce Environment Setup Script
# ============================================================
# Creates a conda environment with all dependencies needed for
# DualForce training, inference, and data preprocessing.
#
# Usage:
#   bash scripts/setup_env.sh [ENV_NAME]
#
# Default env name: dualforce
# ============================================================

set -e

ENV_NAME="${1:-dualforce}"
PYTHON_VERSION="3.10"

echo "=========================================="
echo " DualForce Environment Setup"
echo "=========================================="
echo " Environment name: ${ENV_NAME}"
echo " Python version:   ${PYTHON_VERSION}"
echo "=========================================="

# --------------------------------------------------
# 1. Create conda environment
# --------------------------------------------------
echo ""
echo "[Step 1/6] Creating conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists. Activating..."
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "  Python: $(python --version)"
echo "  Pip:    $(pip --version)"

# --------------------------------------------------
# 2. Install PyTorch (CUDA 12.1)
# --------------------------------------------------
echo ""
echo "[Step 2/6] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# --------------------------------------------------
# 3. Install core dependencies (MOVA + DualForce)
# --------------------------------------------------
echo ""
echo "[Step 3/6] Installing core dependencies..."

# MOVA requirements
pip install diffusers>=0.31.0 transformers>=4.48.0 accelerate>=0.33.0
pip install safetensors ftfy einops mmengine
pip install sentencepiece protobuf

# Flash attention (for efficient attention)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "  Warning: flash-attn install failed (optional)"

# UniCP (Context Parallel, optional)
pip install yunchang 2>/dev/null || echo "  Warning: yunchang install failed (optional)"

# --------------------------------------------------
# 4. Install data preprocessing dependencies
# --------------------------------------------------
echo ""
echo "[Step 4/6] Installing data preprocessing dependencies..."

# Video processing
pip install yt-dlp opencv-python-headless av

# HuBERT audio features
pip install librosa soundfile

# Face detection
pip install mediapipe 2>/dev/null || echo "  Warning: mediapipe install failed"

# DAC audio codec (MOVA audio VAE uses this)
pip install descript-audio-codec 2>/dev/null || echo "  Warning: dac install failed"

# FLAME/EMOCA dependencies (install separately if needed)
# pip install pytorch3d  # Requires special install
# pip install chumpy trimesh  # For FLAME model

# --------------------------------------------------
# 5. Install optional dependencies
# --------------------------------------------------
echo ""
echo "[Step 5/6] Installing optional dependencies..."

# Evaluation
pip install pytorch-fid 2>/dev/null || echo "  Warning: pytorch-fid install failed (needed for evaluation)"
pip install lpips 2>/dev/null || echo "  Warning: lpips install failed (needed for evaluation)"

# Visualization
pip install tensorboard wandb
pip install matplotlib pillow imageio imageio-ffmpeg

# Development
pip install tqdm rich

# --------------------------------------------------
# 6. Install MOVA/DualForce as editable package
# --------------------------------------------------
echo ""
echo "[Step 6/6] Installing DualForce in editable mode..."

# Get the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_ROOT}"
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e . 2>/dev/null || echo "  Note: No setup.py/pyproject.toml found, using PYTHONPATH instead"
else
    echo "  No setup.py found. Add project root to PYTHONPATH:"
    echo "  export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""
fi

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo " Activate environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " Set PYTHONPATH:"
echo "   export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""
echo ""
echo " Verify installation:"
echo "   python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo "   python -c \"from mova.diffusion.models import WanModel; print('MOVA models OK')\""
echo ""
echo " Next steps:"
echo "   1. Verify MOVA inference:    python scripts/verify_mova_inference.py --load-only"
echo "   2. Verify DualForce model:   python scripts/verify_dualforce.py"
echo "   3. Download HDTF data:       python scripts/download/download_hdtf.py --help"
echo ""

# --------------------------------------------------
# Optional: Install LivePortrait (for struct latent extraction)
# --------------------------------------------------
echo " Optional dependencies (install manually if needed):"
echo "   # LivePortrait (for 3D struct latent extraction)"
echo "   git clone https://github.com/KwaiVGI/LivePortrait.git"
echo "   cd LivePortrait && pip install -e ."
echo ""
echo "   # EMOCA (for FLAME parameter extraction)"
echo "   git clone https://github.com/radekd91/emoca.git"
echo "   cd emoca && pip install -e ."
echo ""
echo "   # RetinaFace (for face detection, better than OpenCV)"
echo "   pip install retinaface-pytorch"
echo ""
