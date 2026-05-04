#!/usr/bin/env bash
# watchdog_v2.sh — fresh CW10 run with updated PALR best config
# 7 agents × 5 seeds = 35 runs, 7 batches of 5
#
# Agents:
#   0: SAC-FixedLR
#   1: SAC-ShrinkAndPerturb
#   2: SAC-PeriodicReset
#   3: SAC-L2Reg
#   4: PALR-SAC (best: beta=0.0, sigma=0.5, thresh=0.05)
#   5: PALR-NoPerturb (LR only: beta=0.5, rank_beta=0.5)
#   6: PALR-NoScale (perturb only: sigma=0.5, thresh=0.05)
#
# Run with:
#   nohup bash palr_plasticity_aware_lr/src/watchdog_v2.sh \
#       > palr_plasticity_aware_lr/results/logs/watchdog_v2.log 2>&1 &

REPO="/home/hzhang/work/rlBigWorld"
BASE="bash ${REPO}/run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py"
LOGDIR="${REPO}/palr_plasticity_aware_lr/results/logs"
ARGS="--episodes_per_task 200 --seeds 1 --batch_size 2048"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

launch_agent() {
    local seed=$1 agent=$2
    local suffix="_v2_seed${seed}_agent${agent}"
    nohup $BASE $ARGS \
        --seed_offset $seed --ckpt_suffix $suffix --agent_idx $agent \
        > "${LOGDIR}/v2_seed${seed}_agent${agent}.log" 2>&1 &
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

log "=== watchdog_v2 started: 7 agents × 5 seeds = 35 runs ==="

# Batch 1: seed0 agents 0-4
log "=== Batch 1/7 ==="
launch_agent 0 0; launch_agent 0 1; launch_agent 0 2; launch_agent 0 3; launch_agent 0 4
disown -a; wait_for_batch

# Batch 2: seed0 agents 5-6, seed1 agents 0-2
log "=== Batch 2/7 ==="
launch_agent 0 5; launch_agent 0 6; launch_agent 1 0; launch_agent 1 1; launch_agent 1 2
disown -a; wait_for_batch

# Batch 3: seed1 agents 3-6, seed2 agent 0
log "=== Batch 3/7 ==="
launch_agent 1 3; launch_agent 1 4; launch_agent 1 5; launch_agent 1 6; launch_agent 2 0
disown -a; wait_for_batch

# Batch 4: seed2 agents 1-5
log "=== Batch 4/7 ==="
launch_agent 2 1; launch_agent 2 2; launch_agent 2 3; launch_agent 2 4; launch_agent 2 5
disown -a; wait_for_batch

# Batch 5: seed2 agent 6, seed3 agents 0-3
log "=== Batch 5/7 ==="
launch_agent 2 6; launch_agent 3 0; launch_agent 3 1; launch_agent 3 2; launch_agent 3 3
disown -a; wait_for_batch

# Batch 6: seed3 agents 4-6, seed4 agents 0-1
log "=== Batch 6/7 ==="
launch_agent 3 4; launch_agent 3 5; launch_agent 3 6; launch_agent 4 0; launch_agent 4 1
disown -a; wait_for_batch

# Batch 7: seed4 agents 2-6
log "=== Batch 7/7 ==="
launch_agent 4 2; launch_agent 4 3; launch_agent 4 4; launch_agent 4 5; launch_agent 4 6
disown -a; wait_for_batch

# Merge all v2 checkpoints
log "=== All agents complete! Merging checkpoints... ==="
bash ${REPO}/run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py \
    >> "${LOGDIR}/watchdog_v2.log" 2>&1

log "=== Done. Results in palr_plasticity_aware_lr/results/cw_raw_results.json ==="
