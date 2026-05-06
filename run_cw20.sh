#!/usr/bin/env bash
# =============================================================================
#  run_cw20.sh — Continual-World CW20 launcher (Singularity-aware)
#
#  Runs CW20 experiments inside palr_container.sif, reusing the in-container
#  conda env `palr` (Python 3.9 + mujoco + metaworld + torch + jbw + ...).
#
#  Resume: run_cw_experiments.py auto-resumes from any existing
#          palr_plasticity_aware_lr/results/cw_checkpoint_cw20_seed${S}_agent${A}.json
#          (agent-level granularity — completed agents are skipped).
#
#  Modes:
#    bash run_cw20.sh                    # full sweep: 5 agents × 3 seeds = 15 runs
#    bash run_cw20.sh --single           # one (seed, agent) pair (vars: SEED, AGENT)
#    bash run_cw20.sh --merge            # merge all CW20 checkpoints to a single file
#
#  Override knobs (env vars):
#    SIF             path to palr_container.sif  [default: ${PROJ_DIR}/palr_container.sif]
#    PROJ_DIR        project root                [default: dirname of this script]
#    EPISODES        --episodes_per_task         [default: 200]
#    BATCH_SIZE      --batch_size                [default: 2048]
#    SEEDS           seeds to run (space-sep.)   [default: "0 1 2"]
#    AGENTS          agent indices (space-sep.)  [default: "0 3 4 5 6"]
#                    0=SAC-FixedLR, 3=SAC-L2Reg,
#                    4=PALR-SAC, 5=PALR-NoPerturb, 6=PALR-NoScale
#    PARALLEL        runs per batch              [default: 5  → all agents of one seed]
#    CUDA_VISIBLE_DEVICES                        [default: 0]
#
#  Examples:
#    # Full sweep, picking up from existing checkpoints
#    bash run_cw20.sh
#
#    # Re-run a single hole (e.g. seed2 / PALR-SAC)
#    SEED=2 AGENT=4 bash run_cw20.sh --single
#
#    # Just merge whatever's already in results/
#    bash run_cw20.sh --merge
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SIF="${SIF:-${PROJ_DIR}/palr_container.sif}"
SCRIPT="palr_plasticity_aware_lr/src/run_cw_experiments.py"
MERGE_SCRIPT="palr_plasticity_aware_lr/src/merge_cw_checkpoints.py"
LOGDIR="${PROJ_DIR}/palr_plasticity_aware_lr/results/logs"
mkdir -p "${LOGDIR}"

# ── Knobs ─────────────────────────────────────────────────────────────────────
EPISODES="${EPISODES:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
SEEDS="${SEEDS:-0 1 2}"
AGENTS="${AGENTS:-0 3 4 5 6}"
PARALLEL="${PARALLEL:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[run_cw20] $(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# ── Container runtime ─────────────────────────────────────────────────────────
if command -v apptainer &>/dev/null; then
    SG=apptainer
else
    SG=singularity
fi

if [[ ! -f "${SIF}" ]]; then
    log "ERROR: SIF not found: ${SIF}"
    log "       Build it first:  ${SG} build [--fakeroot|--remote] ${SIF} ${PROJ_DIR}/palr_container.def"
    exit 1
fi

# Wrapper that runs a python script inside the container with all the env
# vars CW20 needs. Args after the script name are forwarded verbatim.
run_in_container() {
    local pyscript="$1"; shift
    ${SG} exec --nv \
        --bind "${PROJ_DIR}:/workspace" \
        "${SIF}" \
        bash -c "
            set -e
            cd /workspace
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
            export MUJOCO_GL=egl
            export PYOPENGL_PLATFORM=egl
            export EGL_PLATFORM=surfaceless
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            export OMP_NUM_THREADS=4
            unset DISPLAY
            python ${pyscript} $*
        "
}

# ── Single (seed, agent) launch ───────────────────────────────────────────────
launch_one() {
    local seed=$1 agent=$2
    local suffix="_cw20_seed${seed}_agent${agent}"
    local logfile="${LOGDIR}/cw20_seed${seed}_agent${agent}.log"

    log "  launching seed=${seed} agent=${agent}  (log: ${logfile})"
    run_in_container "${SCRIPT}" \
        --cw20 \
        --episodes_per_task "${EPISODES}" \
        --seeds 1 \
        --batch_size "${BATCH_SIZE}" \
        --seed_offset "${seed}" \
        --agent_idx "${agent}" \
        --ckpt_suffix "${suffix}" \
        > "${logfile}" 2>&1 &
}

wait_for_batch() {
    log "Waiting for current batch to finish..."
    while pgrep -f "run_cw_experiments.py" > /dev/null 2>&1; do
        sleep 60
    done
    log "Batch done; flushing..."
    sleep 10
}

# ── Subcommands ───────────────────────────────────────────────────────────────
case "${1:-}" in
    --single)
        SEED="${SEED:-0}"
        AGENT="${AGENT:-4}"
        log "Single run: seed=${SEED} agent=${AGENT}"
        launch_one "${SEED}" "${AGENT}"
        wait
        log "Done."
        exit 0
        ;;

    --merge)
        log "Merging CW20 checkpoints..."
        run_in_container "${MERGE_SCRIPT}" --suffix cw20
        log "Done. → palr_plasticity_aware_lr/results/cw20_raw_results.json"
        exit 0
        ;;

    "")
        ;;  # fall through to full sweep

    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--single | --merge]"
        exit 1
        ;;
esac

# ── Full sweep ────────────────────────────────────────────────────────────────
log "=== CW20 sweep: seeds=[${SEEDS}]  agents=[${AGENTS}]  parallel=${PARALLEL} ==="
log "    (existing checkpoints in palr_plasticity_aware_lr/results/ will be reused)"

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

# Drain any remaining background jobs (last partial batch)
wait_for_batch

# ── Merge ─────────────────────────────────────────────────────────────────────
log "=== All runs complete. Merging checkpoints... ==="
run_in_container "${MERGE_SCRIPT}" --suffix cw20 \
    >> "${LOGDIR}/watchdog_cw20.log" 2>&1 || true

log "=== Done. Results in palr_plasticity_aware_lr/results/cw20_raw_results.json ==="
