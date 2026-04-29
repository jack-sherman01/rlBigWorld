#!/usr/bin/env bash


# set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
# All paths are relative to PROJ_DIR; override by setting env vars before
# calling this script (e.g.  SEED=1 bash run_on_HPC.sh).

# Inside a SLURM job, BASH_SOURCE[0] resolves to the spool copy of the script
# (/var/spool/slurm/jobXXX/), not the real project dir.  Use SLURM_SUBMIT_DIR
# (set by sbatch to wherever you ran it) and fall back to dirname only for
# interactive use.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && "${SLURM_SUBMIT_DIR}" != /var/www/* ]]; then
    PROJ_DIR="${PROJ_DIR:-${SLURM_SUBMIT_DIR}}"
else
    PROJ_DIR="${PROJ_DIR:-${_script_dir}}"
fi
# SIF="${SIF:-${PROJ_DIR}/habitat_v3.sif}"       # where the built image lives
DEF="${DEF:-${PROJ_DIR}/habitat_v3.def}"       # Singularity definition file

# Training knobs
SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_ENVS="${NUM_ENVS:-4}"
# CONFIG="${CONFIG:-palr_habitat/configs/ddppo_palr_fetch.yaml}"
CONFIG="${CONFIG:-palr_habitat/configs/ddppo_baseline_fetch.yaml}"
# NOTE: change to ddppo_baseline_fetch.yaml for the baseline run:
# CONFIG="${CONFIG:-palr_habitat/configs/ddppo_baseline_fetch.yaml}"

OUTDIR="${OUTDIR:-results/palr_seed${SEED}}"

# GPU visibility (SLURM sets CUDA_VISIBLE_DEVICES automatically when --gres is
# used; override here only if running outside SLURM).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Weights & Biases
WANDB_PROJECT="${WANDB_PROJECT:-palr-habitat}"
WANDB_ENTITY="${WANDB_ENTITY:-palr-habitat}"
# WANDB_RUN_NAME="${WANDB_RUN_NAME:-palr-ours}" # change to "palr-baseline" for the baseline run
WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline}" # change to "palr-baseline" for the baseline run

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[run_on_HPC] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

module load intel/singularity/singularity-4.2.2

# # ── Singularity / Apptainer availability ──────────────────────────────────────
# # Many HPC clusters provide Singularity or Apptainer as an environment module.
# # Adjust the module name to match your cluster, or remove if already in PATH.
# if ! command -v singularity &>/dev/null && ! command -v apptainer &>/dev/null; then
#     log "singularity/apptainer not in PATH — attempting to load module …"
#     module load singularity 2>/dev/null \
#         || module load apptainer 2>/dev/null \
#         || { log "ERROR: cannot find singularity or apptainer"; exit 1; }
# fi

# Prefer 'apptainer' if available (it is the upstream rename of Singularity).
if command -v apptainer &>/dev/null; then
    SG=apptainer
else
    SG=singularity
fi
log "Using container runtime: ${SG} ($(${SG} --version))"

# ── Sweep mode ────────────────────────────────────────────────────────────────
# Builds the SIF once (if needed), then submits 3 independent SLURM jobs —
# seeds 0, 1, 2 — all for PALR. SLURM runs them in parallel as GPUs free up.
if [[ "${1:-}" == "--sweep" ]]; then
    mkdir -p "${PROJ_DIR}/logs"

    if [[ ! -f "${SIF}" ]]; then
        log "SIF not found — building before sweep …"
        if [[ ! -f "${DEF}" ]]; then
            log "ERROR: definition file not found: ${DEF}"; exit 1
        fi
        ${SG} build --fakeroot "${SIF}" "${DEF}"
        log "Build complete: ${SIF}"
    else
        log "SIF already exists — skipping build: ${SIF}"
    fi

    for seed in 0 1 2; do
        sbatch \
            --job-name="palr_s${seed}" \
            --output="${PROJ_DIR}/logs/palr_seed${seed}_%j.out" \
            --error="${PROJ_DIR}/logs/palr_seed${seed}_%j.err" \
            --export="ALL,SEED=${seed},OUTDIR=results/palr_seed${seed},CONFIG=palr_habitat/configs/ddppo_palr_fetch.yaml" \
            "${BASH_SOURCE[0]}"
        log "Submitted PALR seed=${seed} → results/palr_seed${seed}"
    done
    exit 0
fi

# # ── Step 1: build SIF if absent ──────────────────────────────────────────────
# if [[ ! -f "${SIF}" ]]; then
#     log "SIF not found at: ${SIF}"
#     log "Building from: ${DEF}"

#     if [[ ! -f "${DEF}" ]]; then
#         log "ERROR: definition file not found: ${DEF}"
#         exit 1
#     fi

#     # --fakeroot lets unprivileged users build on most clusters.
#     # Remove it if your site grants real root or uses --sandbox instead.
#     ${SG} build --fakeroot "${SIF}" "${DEF}"
#     log "Build complete: ${SIF}"
# else
#     log "SIF already exists — skipping build: ${SIF}"
# fi

# ── Step 2: prepare output directory ─────────────────────────────────────────
mkdir -p "${PROJ_DIR}/${OUTDIR}/checkpoints"
mkdir -p "${PROJ_DIR}/${OUTDIR}/videos"
mkdir -p "${PROJ_DIR}/logs"

log "Output directory: ${PROJ_DIR}/${OUTDIR}"

# ── Step 3: run training ──────────────────────────────────────────────────────
log "Launching training (seed=${SEED}, gpus=${NUM_GPUS}, envs=${NUM_ENVS}) …"

cd $SLURM_SUBMIT_DIR

container_path=/work/hezhang/hrii/singularity_rlBigWorld/torch2_3.sif

# Note: the following source ~/miniforge3/etc/profile.d/conda.sh only works for Heng.
singularity exec --disable-cache --nv $container_path bash -c "
  source ~/miniforge3/etc/profile.d/conda.sh && \
  conda activate palr_habitat_v3 && \





  cd $PROJ_DIR && \
  export __EGL_VENDOR_LIBRARY_FILENAMES=/work/hezhang/rlBigWorld/nvidia_egl_vendor.json && \
  export LD_LIBRARY_PATH=/.singularity.d/libs:{LD_LIBRARY_PATH} export CUDA_VISIBLE_DEVICES="{CUDA_VISIBLE_DEVICES:-0}" && \
  export PYOPENGL_PLATFORM=egl && \
  export EGL_PLATFORM=surfaceless && \
  unset DISPLAY && \
  export WANDB_MODE=online && \
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} && \
  export HABITAT_SIM_LOG=quiet && \
  export MAGNUM_LOG=quiet && \
  export TF_FORCE_GPU_ALLOW_GROWTH=true && \
  export OMP_NUM_THREADS=14 && \
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


log "Training finished."



# ${SG} exec \
#     --nv \
#     --bind "${PROJ_DIR}":/workspace \
#     "${SIF}" \
#     bash -c "
#         cd /workspace
#         export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
#         export HABITAT_SIM_LOG=quiet
#         export MAGNUM_LOG=quiet
#         export PYOPENGL_PLATFORM=egl
#         export TF_FORCE_GPU_ALLOW_GROWTH=true
#         export OMP_NUM_THREADS=4
#         export WANDB_PROJECT=${WANDB_PROJECT}

#         torchrun \
#             --nproc_per_node=${NUM_GPUS} \
#             --rdzv_backend=c10d \
#             --rdzv_endpoint=localhost:0 \
#             palr_habitat/src/palr_trainer.py \
#                 --config  ${CONFIG} \
#                 --seed    ${SEED} \
#                 --num_envs ${NUM_ENVS} \
#                 --outdir  ${OUTDIR} \
#                 --resume  auto
#     "

# log "Training finished."


# torchrun --standalone --nnodes=1 --nproc_per_node=1 --max_restarts=0 palr_habitat/src/palr_trainer.py --config palr_habitat/configs/ddppo_baseline_fetch.yaml --seed 0 --num_envs 4 --outdir results/debug_71_4_seed0

# export  __EGL_VENDOR_LIBRARY_FILENAMES=/work/hezhang/rlBigWorld/nvidia_egl_vendor.json
# export PYOPENGL_PLATFORM=egl
# export EGL_PLATFORM=surfaceless
# unset DISPLAY
# export HABITAT_SIM_LOG=quiet
# export MAGNUM_LOG=quiet
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# export OMP_NUM_THREADS=14

