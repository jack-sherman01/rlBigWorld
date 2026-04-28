#!/usr/bin/env bash
# =============================================================================
#  palr_habitat/setup_conda_v3.sh
#
#  Creates the "palr_habitat_v3" conda env with habitat 0.3.2.  Kept as a
#  SEPARATE env from the 0.2.5 one (palr_habitat) so we can fall back if
#  0.3.x has a regression.
#
#    - habitat-sim 0.3.2  (EGL headless, Bullet)
#    - habitat-lab 0.3.220241205 + habitat-baselines 0.3.220241205
#    - PyTorch 2.3.1 + CUDA 12.1   (matches driver 580 / CUDA 13 host runtime;
#                                    cu121 wheels are compatible with 12.x+
#                                    driver via NVIDIA forward compat)
#    - python 3.10  (0.3.x supports 3.9–3.11; 3.10 is the upstream-recommended)
#
#  Usage:
#    conda activate base
#    bash setup_conda_v3.sh
# =============================================================================
set -euo pipefail

ENV="palr_habitat_v3"
PY="3.9"

echo "============================================================"
echo "  PALR-Habitat v3 environment setup"
echo "  env name : ${ENV}"
echo "  python   : ${PY}"
echo "  habitat  : 0.3.2"
echo "============================================================"

# --------------------------------------------------------------------------- #
# 1. Conda env
# --------------------------------------------------------------------------- #
if conda env list | grep -q "^${ENV} "; then
    echo "[skip] conda env '${ENV}' already exists — to start fresh:"
    echo "       conda env remove -n ${ENV} -y"
else
    echo ""
    echo "=== Step 1/5  Creating conda env ==="
    conda create -n "${ENV}" python="${PY}" -y
fi

# --------------------------------------------------------------------------- #
# 2. habitat-sim 0.3.2 (conda only)
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 2/5  Installing habitat-sim 0.3.2 (headless EGL + Bullet) ==="
conda install -n "${ENV}" \
    -c aihabitat -c conda-forge \
    habitat-sim=0.3.2 withbullet headless \
    -y

# --------------------------------------------------------------------------- #
# 3. PyTorch 2.3.1 + CUDA 12.1
#    (driver 580 supports CUDA 12.x; cu121 wheels run fine on driver >= 525)
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 3/5  Installing PyTorch 2.3.1 + CUDA 12.1 ==="
conda run -n "${ENV}" pip install \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# --------------------------------------------------------------------------- #
# 4. habitat-lab 0.3.2 / habitat-baselines 0.3.2
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 4/5  Installing habitat-lab 0.3.2 + habitat-baselines 0.3.2 ==="
conda run -n "${ENV}" pip install \
    habitat-lab==0.3.220241205 \
    habitat-baselines==0.3.220241205

# --------------------------------------------------------------------------- #
# 5. Misc Python deps
# --------------------------------------------------------------------------- #
echo ""
echo "=== Step 5/5  Installing Python dependencies ==="
conda run -n "${ENV}" pip install \
    "numpy<2" scipy matplotlib pandas \
    tensorboard \
    imageio "imageio-ffmpeg" \
    tqdm pyyaml gym

# --------------------------------------------------------------------------- #
# Sanity probe
# --------------------------------------------------------------------------- #
echo ""
echo "=== Smoke import ==="
conda run -n "${ENV}" python -c "
import habitat, habitat_sim, torch
print('habitat       :', habitat.__version__)
print('habitat_sim   :', habitat_sim.__version__)
print('torch         :', torch.__version__, 'cuda', torch.version.cuda)
print('cuda available:', torch.cuda.is_available(), 'device count', torch.cuda.device_count())
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate :  conda activate ${ENV}"
echo "  Next step:  bash download_data_v3.sh"
echo "============================================================"
