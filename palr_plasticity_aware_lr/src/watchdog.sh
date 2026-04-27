#!/usr/bin/env bash
# watchdog.sh — re-runs the 26 missing agent-seed combinations
# (seeds 0-3 lost to checkpoint race condition; seed4 crashed mid-run)
#
# Fix applied: per-agent-per-seed checkpoint suffix prevents all race conditions.
# Each agent writes exclusively to cw_checkpoint_seed{N}_agent{M}.json.
#
# Batches of 5 to keep RAM usage ~2.5 GB (safe on 30 GB machine).
#
# Run with: nohup bash watchdog.sh > results/logs/watchdog_rerun.log 2>&1 &

REPO="/home/hzhang/work/rlBigWorld"
BASE="bash ${REPO}/run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py"
LOGDIR="${REPO}/palr_plasticity_aware_lr/results/logs"
ARGS="--episodes_per_task 200 --seeds 1 --batch_size 2048"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

launch_agent() {
    local seed=$1 agent=$2
    local suffix="_seed${seed}_agent${agent}"
    nohup $BASE $ARGS \
        --seed_offset $seed --ckpt_suffix $suffix --agent_idx $agent \
        > "${LOGDIR}/seed${seed}_agent${agent}.log" 2>&1 &
    log "  launched seed${seed} agent${agent} (pid=$!)"
}

wait_for_batch() {
    log "Waiting for current batch to finish..."
    while pgrep -f "run_cw_experiments.py" > /dev/null 2>&1; do
        sleep 120
    done
    log "All processes finished. Sleeping 30s for files to flush..."
    sleep 30
}

# ── 26 missing agents, 5 per batch ───────────────────────────────────────────
# seed0 missing: a0,1,2,3,5  (a4 and a6 already saved)
# seed1 missing: a0,1,2,4,5  (a3 and a6 already saved)
# seed2 missing: a0,1,2,6    (a3, a4, a5 already saved)
# seed3 missing: a0,1,2,3,5  (a4 and a6 already saved)
# seed4 missing: a0-6        (all — crashed with no checkpoint)

BATCH_A=("0 0"  "0 1"  "0 2"  "0 3"  "0 5")
BATCH_B=("1 0"  "1 1"  "1 2"  "1 4"  "1 5")
BATCH_C=("2 0"  "2 1"  "2 2"  "2 6"  "3 0")
BATCH_D=("3 1"  "3 2"  "3 3"  "3 5"  "4 0")
BATCH_E=("4 1"  "4 2"  "4 3"  "4 4"  "4 5")
BATCH_F=("4 6")

log "=== Watchdog started. Launching batch-A (${#BATCH_A[@]} agents) ==="
for pair in "${BATCH_A[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

log "=== Launching batch-B (${#BATCH_B[@]} agents) ==="
for pair in "${BATCH_B[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

log "=== Launching batch-C (${#BATCH_C[@]} agents) ==="
for pair in "${BATCH_C[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

log "=== Launching batch-D (${#BATCH_D[@]} agents) ==="
for pair in "${BATCH_D[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

log "=== Launching batch-E (${#BATCH_E[@]} agents) ==="
for pair in "${BATCH_E[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

log "=== Launching batch-F (${#BATCH_F[@]} agent) ==="
for pair in "${BATCH_F[@]}"; do launch_agent $pair; done
disown -a
wait_for_batch

# ── Merge all checkpoints (new + existing) ────────────────────────────────────
log "=== All agents complete! Merging checkpoints... ==="
bash ${REPO}/run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py \
    >> "${LOGDIR}/watchdog_rerun.log" 2>&1

log "=== Done. Results in palr_plasticity_aware_lr/results/cw_raw_results.json ==="
