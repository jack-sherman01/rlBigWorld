#!/usr/bin/env bash
# watchdog_cw20.sh — CW20 run (20 tasks, 200 ep/task = 4000 eps per agent)
# 5 agents × 5 seeds = 25 runs, 5 batches of 5
#
# Agents (no reset-based methods):
#   0: SAC-FixedLR
#   1: SAC-L2Reg          (was index 3 in CW10; re-indexed for CW20)
#   2: PALR-SAC (ours)    (best config: beta=0.0, sigma=0.5, thresh=0.05)
#   3: PALR-NoPerturb
#   4: PALR-NoScale
#
# NOTE: --agent_idx maps to make_agents() order in run_cw_experiments.py:
#   0=SAC-FixedLR, 1=SAC-ShrinkAndPerturb (skip), 2=SAC-PeriodicReset (skip),
#   3=SAC-L2Reg, 4=PALR-SAC, 5=PALR-NoPerturb, 6=PALR-NoScale
# We launch only agents 0, 3, 4, 5, 6.
#
# Run with:
#   nohup bash palr_plasticity_aware_lr/src/watchdog_cw20.sh \
#       > palr_plasticity_aware_lr/results/logs/watchdog_cw20.log 2>&1 &

REPO="/home/hzhang/work/rlBigWorld"
BASE="bash ${REPO}/run.sh palr_plasticity_aware_lr/src/run_cw_experiments.py"
LOGDIR="${REPO}/palr_plasticity_aware_lr/results/logs"
ARGS="--episodes_per_task 200 --seeds 1 --batch_size 2048 --cw20"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

launch_agent() {
    local seed=$1 agent=$2
    local suffix="_cw20_seed${seed}_agent${agent}"
    nohup $BASE $ARGS \
        --seed_offset $seed --ckpt_suffix $suffix --agent_idx $agent \
        > "${LOGDIR}/cw20_seed${seed}_agent${agent}.log" 2>&1 &
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

log "=== watchdog_cw20 started: 5 agents × 3 seeds = 15 runs ==="
log "=== Agents: 0=SAC-FixedLR, 3=SAC-L2Reg, 4=PALR-SAC, 5=PALR-NoPerturb, 6=PALR-NoScale ==="

# Batch 1: seed0 — all 5 agents
log "=== Batch 1/3: seed0 ==="
launch_agent 0 0; launch_agent 0 3; launch_agent 0 4; launch_agent 0 5; launch_agent 0 6
disown -a; wait_for_batch

# Batch 2: seed1 — all 5 agents
log "=== Batch 2/3: seed1 ==="
launch_agent 1 0; launch_agent 1 3; launch_agent 1 4; launch_agent 1 5; launch_agent 1 6
disown -a; wait_for_batch

# Batch 3: seed2 — all 5 agents
log "=== Batch 3/3: seed2 ==="
launch_agent 2 0; launch_agent 2 3; launch_agent 2 4; launch_agent 2 5; launch_agent 2 6
disown -a; wait_for_batch

# Merge all CW20 checkpoints
log "=== All agents complete! Merging checkpoints... ==="
bash ${REPO}/run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py \
    --suffix cw20 \
    >> "${LOGDIR}/watchdog_cw20.log" 2>&1

log "=== Done. Results in palr_plasticity_aware_lr/results/cw20_raw_results.json ==="
