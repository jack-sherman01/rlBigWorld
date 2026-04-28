#!/usr/bin/env bash
# =============================================================================
#  run_on_HPC.sh
#
#  1. Build habitat_v3.sif from habitat_v3.def if the SIF does not yet exist.
#  2. Run PALR DD-PPO Fetch training inside the container.
#
#  Submit to SLURM (single run):
#    sbatch run_on_HPC.sh
#  Or run interactively (after grabbing a GPU node):
#    bash run_on_HPC.sh
#
#  Submit 3 parallel PALR jobs (seeds 0/1/2) — builds SIF first if needed:
#    bash run_on_HPC.sh --sweep
# =============================================================================

# ── SLURM directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=palr_habitat_v3
#SBATCH --output=logs/palr_%j.out
#SBATCH --error=logs/palr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Uncomment and adjust the partition / account for your cluster:
#SBATCH --partition=gpua
#SBATCH --account=heng.zhang@iit.it

set -euo pipefail

# ── Configurable paths ────────────────────────────────────────────────────────
# All paths are relative to PROJ_DIR; override by setting env vars before
# calling this script (e.g.  SEED=1 bash run_on_HPC.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${PROJ_DIR:-${SCRIPT_DIR}}"          # project root (bound to /workspace)
SIF="${SIF:-${PROJ_DIR}/habitat_v3.sif}"       # where the built image lives
DEF="${DEF:-${PROJ_DIR}/habitat_v3.def}"       # Singularity definition file

# Training knobs
SEED="${SEED:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_ENVS="${NUM_ENVS:-16}"
CONFIG="${CONFIG:-palr_habitat/configs/ddppo_palr_fetch.yaml}"
# NOTE: change to ddppo_baseline_fetch.yaml for the baseline run:
# CONFIG="${CONFIG:-palr_habitat/configs/ddppo_baseline_fetch.yaml}"

OUTDIR="${OUTDIR:-results/palr_seed${SEED}}"

# GPU visibility (SLURM sets CUDA_VISIBLE_DEVICES automatically when --gres is
# used; override here only if running outside SLURM).
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Weights & Biases
WANDB_PROJECT="${WANDB_PROJECT:-palr-habitat}"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[run_on_HPC] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# ── Singularity / Apptainer availability ──────────────────────────────────────
# Many HPC clusters provide Singularity or Apptainer as an environment module.
# Adjust the module name to match your cluster, or remove if already in PATH.
if ! command -v singularity &>/dev/null && ! command -v apptainer &>/dev/null; then
    log "singularity/apptainer not in PATH — attempting to load module …"
    module load singularity 2>/dev/null \
        || module load apptainer 2>/dev/null \
        || { log "ERROR: cannot find singularity or apptainer"; exit 1; }
fi

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

# ── Step 1: build SIF if absent ──────────────────────────────────────────────
if [[ ! -f "${SIF}" ]]; then
    log "SIF not found at: ${SIF}"
    log "Building from: ${DEF}"

    if [[ ! -f "${DEF}" ]]; then
        log "ERROR: definition file not found: ${DEF}"
        exit 1
    fi

    # --fakeroot lets unprivileged users build on most clusters.
    # Remove it if your site grants real root or uses --sandbox instead.
    ${SG} build --fakeroot "${SIF}" "${DEF}"
    log "Build complete: ${SIF}"
else
    log "SIF already exists — skipping build: ${SIF}"
fi

# ── Step 2: prepare output directory ─────────────────────────────────────────
mkdir -p "${PROJ_DIR}/${OUTDIR}/checkpoints"
mkdir -p "${PROJ_DIR}/${OUTDIR}/videos"
mkdir -p "${PROJ_DIR}/logs"

log "Output directory: ${PROJ_DIR}/${OUTDIR}"

# ── Step 3: run training ──────────────────────────────────────────────────────
log "Launching training (seed=${SEED}, gpus=${NUM_GPUS}, envs=${NUM_ENVS}) …"

${SG} exec \
    --nv \
    --bind "${PROJ_DIR}":/workspace \
    "${SIF}" \
    bash -c "
        cd /workspace
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
        export HABITAT_SIM_LOG=quiet
        export MAGNUM_LOG=quiet
        export PYOPENGL_PLATFORM=egl
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export OMP_NUM_THREADS=4
        export WANDB_PROJECT=${WANDB_PROJECT}

        torchrun \
            --nproc_per_node=${NUM_GPUS} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=localhost:0 \
            palr_habitat/src/palr_trainer.py \
                --config  ${CONFIG} \
                --seed    ${SEED} \
                --num_envs ${NUM_ENVS} \
                --outdir  ${OUTDIR}
    "

log "Training finished."
