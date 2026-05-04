#!/usr/bin/env bash
# **** Usage: apply for GPU resources on the cluster, then run this script to launch training.
# Run a single training run:
# method includes: baseline, palr, palr_lr_only, palr_per_perturb_only
# bash run_interactive.sh --method palr --seed 0
# bash run_interactive.sh --method palr --seed 1
# bash run_interactive.sh --method palr --seed 2
# **** Usage: Run a parallel sweep over seeds 0, 1, 2 (each on a different GPU):
# bash run_interactive.sh --method palr --sweep


set -euo pipefail

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ── Argument parsing ──────────────────────────────────────────────────────────
METHOD="baseline" # default method

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --sweep)
            SWEEP=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Training knobs
SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_ENVS="${NUM_ENVS:-4}"
CONFIG="${CONFIG:-palr_habitat/configs/ddppo_${METHOD}_fetch.yaml}"
OUTDIR="${OUTDIR:-results/${METHOD}_seed${SEED}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Weights & Biases
WANDB_PROJECT="${WANDB_PROJECT:-palr-habitat}"
WANDB_ENTITY="${WANDB_ENTITY:-palr-habitat}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${METHOD}}"

container_path=/work/hezhang/hrii/singularity_rlBigWorld/torch2_3.sif

log() { echo "[run_local] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# ── Parallel sweep mode ───────────────────────────────────────────────────────
if [[ "${SWEEP:-0}" == "1" ]]; then
    mkdir -p "${PROJ_DIR}/logs"

    for seed in 0 1 2; do
        gpu="${seed}"

        log "===== Starting ${METHOD} seed=${seed} on GPU ${gpu} ====="

        SEED="${seed}" \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        NUM_GPUS=1 \
        CONFIG="palr_habitat/configs/ddppo_${METHOD}_fetch.yaml" \
        OUTDIR="results/${METHOD}_seed${seed}" \
        WANDB_RUN_NAME="${METHOD}_seed${seed}" \
            bash "${BASH_SOURCE[0]}" \
            2>&1 | tee "${PROJ_DIR}/logs/${METHOD}_seed${seed}.log" &
    done

    wait
    log "Sweep complete."
    exit 0
fi

# ── Single run ────────────────────────────────────────────────────────────────
mkdir -p "${PROJ_DIR}/${OUTDIR}/checkpoints"
mkdir -p "${PROJ_DIR}/${OUTDIR}/videos"
mkdir -p "${PROJ_DIR}/logs"

log "Launching training: method=${METHOD} seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES} gpus=${NUM_GPUS}"
log "Config: ${CONFIG}"
log "Output: ${PROJ_DIR}/${OUTDIR}"

singularity exec --disable-cache --nv "${container_path}" bash -c "
  source ~/miniforge3/etc/profile.d/conda.sh && \
  conda activate palr_habitat_v3 && \
  cd ${PROJ_DIR} && \
  export __EGL_VENDOR_LIBRARY_FILENAMES=/work/hezhang/rlBigWorld/nvidia_egl_vendor.json && \
  export LD_LIBRARY_PATH=/.singularity.d/libs:\${LD_LIBRARY_PATH:-} && \
  export PYOPENGL_PLATFORM=egl && \
  export EGL_PLATFORM=surfaceless && \
  unset DISPLAY && \
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
  export HABITAT_SIM_LOG=quiet && \
  export MAGNUM_LOG=quiet && \
  export TF_FORCE_GPU_ALLOW_GROWTH=true && \
  export OMP_NUM_THREADS=14 && \
  export WANDB_MODE=online && \
  export WANDB_PROJECT=${WANDB_PROJECT} && \
  export WANDB_ENTITY=${WANDB_ENTITY} && \
  export WANDB_RUN_NAME=${WANDB_RUN_NAME} && \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    palr_habitat/src/palr_trainer.py \
      --config ${CONFIG} \
      --seed ${SEED} \
      --num_envs ${NUM_ENVS} \
      --outdir ${OUTDIR} \
      --resume auto
"

log "Training finished (method=${METHOD}, seed=${SEED})."
