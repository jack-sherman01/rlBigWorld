#!/usr/bin/env bash
# =============================================================================
#  palr_habitat/setup_conda.sh
#
#  Creates the "palr_habitat" conda environment with:
#    - habitat-sim 0.2.5  (EGL headless, Bullet physics) via conda
#    - habitat-lab 0.2.5 + habitat-baselines 0.2.5  via pip
#    - PyTorch 2.1.2 + CUDA 11.8  via pip
#    - torchvision, scipy, matplotlib, tensorboard
#
#  Tested target: Ubuntu 20.04/22.04, 4× NVIDIA A40 (CUDA 11.8 driver)
#
#  Usage:
#    conda activate base
#    bash setup_conda.sh
# =============================================================================
set -euo pipefail

ENV="palr_habitat"
PY="3.9"

echo "============================================================"
echo "  PALR-Habitat environment setup"
echo "  env name : ${ENV}"
echo "  python   : ${PY}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# 1. Conda env
# --------------------------------------------------------------------------- #
if conda env list | grep -q "^${ENV} "; then
    echo "[skip] conda env '${ENV}' already exists — remove it first if you want a clean install:"
    echo "       conda env remove -n ${ENV}"
else
    echo ""
    echo "=== Step 1/5  Creating conda env ==="
    conda create -n "${ENV}" python="${PY}" -y
fi

# --------------------------------------------------------------------------- #
# 2. habitat-sim (MUST come from conda — no pip wheel for Linux)
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 2/5  Installing habitat-sim 0.2.5 (headless EGL + Bullet) ==="
conda install -n "${ENV}" \
    -c aihabitat -c conda-forge \
    habitat-sim=0.2.5 withbullet headless \
    -y

# --------------------------------------------------------------------------- #
# 3. PyTorch (CUDA 11.8)
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 3/5  Installing PyTorch 2.1.2 + CUDA 11.8 ==="
conda run -n "${ENV}" pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------------------------------------------------------------- #
# 4. habitat-lab / habitat-baselines
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 4/5  Installing habitat-lab 0.2.5 + habitat-baselines 0.2.5 ==="
conda run -n "${ENV}" pip install \
    habitat-lab==0.2.5 \
    habitat-baselines==0.2.5

# --------------------------------------------------------------------------- #
# 5. Misc Python deps
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 5/5  Installing Python dependencies ==="
conda run -n "${ENV}" pip install \
    numpy scipy matplotlib pandas \
    tensorboard \
    imageio "imageio-ffmpeg" \
    tqdm

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Activate :  conda activate ${ENV}"
echo "  Next step:  bash download_data.sh"
echo "============================================================"
