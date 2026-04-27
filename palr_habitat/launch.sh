#!/usr/bin/env bash
# =============================================================================
#  palr_habitat/launch.sh
#
#  Launch PALR or baseline DD-PPO training on 4× A40 GPUs.
#
#  Usage:
#    conda activate palr_habitat
#
#    # PALR agent (our method), 1 seed on all 4 GPUs
#    bash launch.sh --agent palr --seed 0
#
#    # Baseline (fixed LR), seed 1
#    bash launch.sh --agent baseline --seed 1
#
#    # Full 5-seed sweep — launch in 5 separate tmux windows
#    for s in 0 1 2 3 4; do
#        tmux new-window -n "seed${s}" "bash launch.sh --agent palr --seed ${s}"
#    done
#
#  Output:
#    results/palr_seed0/   (tensorboard + checkpoint + metrics.json)
#    results/baseline_seed0/
#
#  Note: each run occupies all 4 GPUs. If running multi-seed concurrently,
#  you need 4 GPUs per seed (i.e., 20 A40s for 5 seeds in parallel).
#  To run sequentially instead, omit the loop above.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/src"
CFG="${SCRIPT_DIR}/configs"

# --------------------------------------------------------------------------- #
# EGL vendor selection (force NVIDIA backend for habitat-sim).
# Some clusters only ship 50_mesa.json under /usr/share/glvnd/egl_vendor.d/,
# which causes habitat-sim to fail with EGL_BAD_PARAMETER on headless GPU
# nodes. We provide a user-local 10_nvidia.json and point libglvnd at it.
# --------------------------------------------------------------------------- #
NVIDIA_EGL_JSON="${HOME}/.local/share/glvnd/egl_vendor.d/10_nvidia.json"
if [[ ! -f "${NVIDIA_EGL_JSON}" ]]; then
    mkdir -p "$(dirname "${NVIDIA_EGL_JSON}")"
    cat > "${NVIDIA_EGL_JSON}" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-${NVIDIA_EGL_JSON}}"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
AGENT="palr"          # palr | baseline
SEED=0
NUM_GPUS=4
ENVS_PER_GPU=16       # 64 total parallel envs
STEPS_TOTAL=50000000  # 50 M env steps per phase (200 M total across 4 phases)

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)    AGENT="$2";     shift 2 ;;
        --seed)     SEED="$2";      shift 2 ;;
        --gpus)     NUM_GPUS="$2";  shift 2 ;;
        --envs)     ENVS_PER_GPU="$2"; shift 2 ;;
        --steps)    STEPS_TOTAL="$2";  shift 2 ;;
        *) echo "Unknown flag: $1" && exit 1 ;;
    esac
done

# --------------------------------------------------------------------------- #
# Config selection
# --------------------------------------------------------------------------- #
if [[ "${AGENT}" == "palr" ]]; then
    CONFIG="${CFG}/ddppo_palr_fetch.yaml"
else
    CONFIG="${CFG}/ddppo_baseline_fetch.yaml"
fi

OUTDIR="${SCRIPT_DIR}/results/${AGENT}_seed${SEED}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "  PALR-Habitat Fetch Rearrangement"
echo "  agent        : ${AGENT}"
echo "  seed         : ${SEED}"
echo "  GPUs         : ${NUM_GPUS}"
echo "  envs/GPU     : ${ENVS_PER_GPU}  (total: $((NUM_GPUS * ENVS_PER_GPU)))"
echo "  steps/phase  : ${STEPS_TOTAL}"
echo "  config       : ${CONFIG}"
echo "  output       : ${OUTDIR}"
echo "============================================================"

export GLOG_minloglevel=2           # suppress habitat C++ logs
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
export MUJOCO_GL="egl"              # headless EGL (no display required)

# --------------------------------------------------------------------------- #
# torchrun launch (DD-PPO across 4 GPUs)
# --------------------------------------------------------------------------- #
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=$((12000 + SEED)) \
    "${SRC}/palr_trainer.py" \
    --config    "${CONFIG}" \
    --seed      "${SEED}" \
    --num_envs  "${ENVS_PER_GPU}" \
    --steps     "${STEPS_TOTAL}" \
    --outdir    "${OUTDIR}" \
    2>&1 | tee "${OUTDIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Done!  Results: ${OUTDIR}"
echo "TensorBoard: tensorboard --logdir ${OUTDIR}/tb"
