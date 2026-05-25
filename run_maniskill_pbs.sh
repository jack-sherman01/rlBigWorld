#!/usr/bin/env bash
# =============================================================================
#  run_maniskill_pbs.sh — PBS job launcher for ManiSkill ViT PALR experiments
#
#  GPU access on this machine requires a PBS job (group pbs_gpu is only
#  granted during job execution). This script submits the full sweep as
#  parallel PBS jobs, or can be used for interactive GPU testing.
#
#  Usage:
#    # Submit full sweep (6 agents × 3 seeds = 18 jobs)
#    bash run_maniskill_pbs.sh --sweep
#
#    # Submit single (seed, agent) pair
#    SEED=0 AGENT=3 bash run_maniskill_pbs.sh --single
#
#    # Interactive GPU session for fast test
#    bash run_maniskill_pbs.sh --interactive
#
#    # Run within an already-active PBS job (called from PBS script)
#    bash run_maniskill_pbs.sh --run
# =============================================================================

set -euo pipefail

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SIF="${SIF:-${PROJ_DIR}/maniskill_vit.sif}"
SCRIPT="maniskill_vit/src/run_experiments.py"
LOGDIR="${PROJ_DIR}/maniskill_vit/logs"
mkdir -p "${LOGDIR}"

# PBS job knobs
WALLTIME="${WALLTIME:-08:00:00}"
NCPUS="${NCPUS:-8}"
MEM="${MEM:-28gb}"
QUEUE="${QUEUE:-gpu}"   # adjust to your cluster's GPU queue name

SEEDS="${SEEDS:-0 1 2}"
AGENTS="${AGENTS:-0 1 2 3 4 5}"
EPISODES="${EPISODES:-400}"
TASK_EPS="${TASK_EPS:-100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[pbs] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# ---------------------------------------------------------------------------
run_in_container() {
    local pyscript="$1"; shift
    apptainer exec --nv \
        --bind "${PROJ_DIR}:/workspace" \
        --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
        "${SIF}" \
        bash -c "
            set -e
            cd /workspace
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
            export MUJOCO_GL=egl
            export PYOPENGL_PLATFORM=egl
            export EGL_PLATFORM=surfaceless
            export SAPIEN_HEADLESS=1
            export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            export OMP_NUM_THREADS=4
            unset DISPLAY
            python ${pyscript} $*
        "
}

# ---------------------------------------------------------------------------
case "${1:-}" in

    --sweep)
        log "Submitting full sweep: seeds=[${SEEDS}] agents=[${AGENTS}]"
        for seed in ${SEEDS}; do
            for agent in ${AGENTS}; do
                logfile="${LOGDIR}/pbs_seed${seed}_agent${agent}.log"
                job_name="ms_s${seed}_a${agent}"
                qsub -N "${job_name}" \
                    -o "${logfile}" -e "${logfile}" \
                    -l "select=1:ncpus=${NCPUS}:ngpus=1:mem=${MEM}" \
                    -l "walltime=${WALLTIME}" \
                    -q "${QUEUE}" \
                    -- bash "${BASH_SOURCE[0]}" --run \
                       "SEED=${seed}" "AGENT=${agent}"
                log "  submitted seed=${seed} agent=${agent} → ${job_name}"
            done
        done
        log "All jobs submitted. Monitor: qstat -u ${USER}"
        exit 0
        ;;

    --single)
        SEED="${SEED:-0}"
        AGENT="${AGENT:-3}"
        logfile="${LOGDIR}/pbs_seed${SEED}_agent${AGENT}.log"
        qsub -N "ms_s${SEED}_a${AGENT}" \
            -o "${logfile}" -e "${logfile}" \
            -l "select=1:ncpus=${NCPUS}:ngpus=1:mem=${MEM}" \
            -l "walltime=${WALLTIME}" \
            -q "${QUEUE}" \
            -- bash "${BASH_SOURCE[0]}" --run "SEED=${SEED}" "AGENT=${AGENT}"
        log "Submitted seed=${SEED} agent=${AGENT}. Monitor: qstat -u ${USER}"
        exit 0
        ;;

    --interactive)
        log "Starting interactive PBS session with GPU..."
        log "  Once inside: bash run_maniskill.sh --fast"
        qsub -I \
            -l "select=1:ncpus=${NCPUS}:ngpus=1:mem=${MEM}" \
            -l "walltime=02:00:00" \
            -q "${QUEUE}"
        exit 0
        ;;

    --run)
        # Called from within a PBS job — remaining args are VAR=VALUE pairs
        for kv in "${@:2}"; do export "${kv}"; done
        SEED="${SEED:-0}"
        AGENT="${AGENT:-3}"
        log "Running inside PBS job: seed=${SEED} agent=${AGENT}"
        if [[ ! -f "${SIF}" ]]; then
            log "ERROR: SIF not found: ${SIF}"; exit 1
        fi
        run_in_container "${SCRIPT}" \
            --seeds 1 \
            --seed_offset "${SEED}" \
            --agent_idx "${AGENT}" \
            --episodes "${EPISODES}" \
            --task_episodes "${TASK_EPS}"
        log "Done."
        exit 0
        ;;

    *)
        echo "Usage: $0 [--sweep | --single | --interactive | --run]"
        echo "  --sweep        Submit all seeds × agents as PBS jobs"
        echo "  --single       Submit one (SEED, AGENT) PBS job"
        echo "  --interactive  Start interactive PBS session with GPU"
        echo "  --run          Execute inside an active PBS job"
        exit 1
        ;;
esac
