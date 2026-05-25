#!/usr/bin/env bash
# =============================================================================
#  run_maniskill.sh — ManiSkill + ViT PALR experiment launcher
#
#  Runs inside maniskill_vit.sif (Singularity/Apptainer).
#  4-task Heterogeneous Skill Stream: PickCube → StackCube → TurnFaucet → PushCube
#  ViT-Small from scratch + SAC + PALR / baselines.
#
#  Modes:
#    bash run_maniskill.sh                  # full sweep: 6 agents × 3 seeds
#    bash run_maniskill.sh --single         # one (seed, agent) pair
#    bash run_maniskill.sh --fast           # 50-episode debug run
#    bash run_maniskill.sh --build          # build maniskill_vit.sif container
#
#  Override knobs (env vars):
#    SIF          path to maniskill_vit.sif  [default: ${PROJ_DIR}/maniskill_vit.sif]
#    SEEDS        seed range (space-sep.)    [default: "0 1 2"]
#    AGENTS       agent indices              [default: "0 1 2 3 4 5"]
#                 0=FixedLR, 1=L2Reg, 2=ShrinkPerturb
#                 3=PALR-SAC, 4=PALR-NoPerturb, 5=PALR-NoScale
#    EPISODES     total episodes             [default: 400]
#    TASK_EPS     episodes per task          [default: 100]
#    PARALLEL     runs per batch             [default: 3]
#    CUDA_VISIBLE_DEVICES                   [default: 0]
#
#  Examples:
#    # Full sweep
#    bash run_maniskill.sh
#
#    # Debug one run
#    bash run_maniskill.sh --fast
#
#    # Single hole (seed=1, PALR agent)
#    SEED=1 AGENT=3 bash run_maniskill.sh --single
#
#    # Build container first
#    bash run_maniskill.sh --build
# =============================================================================

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SIF="${SIF:-${PROJ_DIR}/maniskill_vit.sif}"
DEF="${PROJ_DIR}/maniskill_vit.def"
SCRIPT="maniskill_vit/src/run_experiments.py"
LOGDIR="${PROJ_DIR}/maniskill_vit/logs"
mkdir -p "${LOGDIR}"

# Knobs
EPISODES="${EPISODES:-400}"
TASK_EPS="${TASK_EPS:-100}"
SEEDS="${SEEDS:-0 1 2}"
AGENTS="${AGENTS:-0 1 2 3 4 5}"
PARALLEL="${PARALLEL:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[run_maniskill] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# Container runtime
if command -v apptainer &>/dev/null; then
    SG=apptainer
else
    SG=singularity
fi

# ---------------------------------------------------------------------------
run_in_container() {
    local pyscript="$1"; shift

    # Bind host Vulkan ICD so SAPIEN can find the NVIDIA Vulkan driver
    local VULKAN_BIND=""
    for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
        if [[ -d "${icd_dir}" ]]; then
            VULKAN_BIND="${VULKAN_BIND} --bind ${icd_dir}:${icd_dir}:ro"
        fi
    done

    ${SG} exec --nv \
        --bind "${PROJ_DIR}:/workspace" \
        ${VULKAN_BIND} \
        "${SIF}" \
        bash -c "
            set -e
            cd /workspace
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
            export MUJOCO_GL=egl
            export PYOPENGL_PLATFORM=egl
            export EGL_PLATFORM=surfaceless
            export SAPIEN_HEADLESS=1
            # Point Vulkan loader to NVIDIA ICD from host
            export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            export OMP_NUM_THREADS=4
            unset DISPLAY
            python ${pyscript} $*
        "
}

launch_one() {
    local seed=$1 agent=$2
    local logfile="${LOGDIR}/seed${seed}_agent${agent}.log"
    log "  launching seed=${seed} agent=${agent}  (log: ${logfile})"
    run_in_container "${SCRIPT}" \
        --seeds 1 \
        --seed_offset "${seed}" \
        --agent_idx "${agent}" \
        --episodes "${EPISODES}" \
        --task_episodes "${TASK_EPS}" \
        > "${logfile}" 2>&1 &
}

wait_for_batch() {
    log "Waiting for current batch to finish..."
    while pgrep -f "run_experiments.py" > /dev/null 2>&1; do
        sleep 30
    done
    log "Batch done."
    sleep 5
}

# ---------------------------------------------------------------------------
case "${1:-}" in
    --build)
        log "Building maniskill_vit.sif from ${DEF} ..."
        if [[ ! -f "${DEF}" ]]; then
            log "ERROR: ${DEF} not found"
            exit 1
        fi
        ${SG} build --fakeroot "${SIF}" "${DEF}" \
            | tee "${PROJ_DIR}/maniskill_vit_build.log"
        log "Build complete → ${SIF}"
        exit 0
        ;;

    --single)
        SEED="${SEED:-0}"
        AGENT="${AGENT:-3}"
        log "Single run: seed=${SEED} agent=${AGENT}"
        if [[ ! -f "${SIF}" ]]; then
            log "ERROR: SIF not found: ${SIF}  — run:  bash run_maniskill.sh --build"
            exit 1
        fi
        launch_one "${SEED}" "${AGENT}"
        wait
        log "Done."
        exit 0
        ;;

    --fast)
        log "Fast debug run (50 episodes, 1 seed, mock env — CPU only, no GPU needed)"
        if [[ ! -f "${SIF}" ]]; then
            log "ERROR: SIF not found: ${SIF}  — run:  bash run_maniskill.sh --build"
            exit 1
        fi
        # No --nv flag: mock mode is CPU-only, avoids GPU device permission hang
        apptainer exec \
            --bind "${PROJ_DIR}:/workspace" \
            "${SIF}" \
            bash -c "
                set -e
                cd /workspace
                export OMP_NUM_THREADS=2
                export PYTHONUNBUFFERED=1
                python ${SCRIPT} --fast --mock
            "
        log "Done."
        exit 0
        ;;

    "")
        ;;

    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--build | --single | --fast]"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------
if [[ ! -f "${SIF}" ]]; then
    log "ERROR: SIF not found: ${SIF}"
    log "       Build it first:  bash run_maniskill.sh --build"
    exit 1
fi

log "=== Full sweep: seeds=[${SEEDS}]  agents=[${AGENTS}]  parallel=${PARALLEL} ==="
log "    results → maniskill_vit/results/"

count=0
for seed in ${SEEDS}; do
    log "=== seed=${seed} ==="
    for agent in ${AGENTS}; do
        launch_one "${seed}" "${agent}"
        count=$((count + 1))
        if (( count % PARALLEL == 0 )); then
            wait_for_batch
        fi
    done
done

wait_for_batch

log "=== All runs complete. Results in maniskill_vit/results/ ==="
