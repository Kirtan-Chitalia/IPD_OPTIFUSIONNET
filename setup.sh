#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for OptiFusionNet
#
# Usage:
#   bash setup.sh              # installs for CUDA 11.8 (default)
#   CUDA_VER=cu121 bash setup.sh   # installs for CUDA 12.1
#
# Requires: Python 3.8+ and pip already on PATH
# =============================================================================

set -euo pipefail

CUDA_VER="${CUDA_VER:-cu118}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_VER}"

echo "============================================================"
echo " OptiFusionNet – Environment Setup"
echo " CUDA wheel index : ${TORCH_INDEX}"
echo "============================================================"

# ── 1. Upgrade pip ──────────────────────────────────────────────
echo ""
echo "[1/3] Upgrading pip..."
pip install --upgrade pip

# ── 2. Install PyTorch (GPU) ────────────────────────────────────
echo ""
echo "[2/3] Installing PyTorch stack for ${CUDA_VER}..."
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

# ── 3. Install remaining dependencies ───────────────────────────
echo ""
echo "[3/3] Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================================"
echo " Setup complete!"
echo " Run 'jupyter notebook notebooks/OptifusionNet.ipynb' to start."
echo "============================================================"
