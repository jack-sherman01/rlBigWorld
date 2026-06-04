#!/usr/bin/env bash
# Resume PALR-SAC (agent 3) and PALR-NoPerturb (agent 4) for seeds 0,1,2
# Runs on A6000 GPU via run.sh; checkpoints auto-saved after each task switch.
#
# Usage:  bash maniskill_vit/run_resume.sh
#         Logs go to maniskill_vit/logs/resume_agent{3,4}_seed{0,1,2}.log

set -euo pipefail

REPO=/home/hzhang/work/rlBigWorld
RUNNER="$REPO/run.sh"
SCRIPT="$REPO/maniskill_vit/src/run_experiments.py"
LOGDIR="$REPO/maniskill_vit/logs"

EPISODES=400        # full 4-task stream (100 eps per task)
TASK_EPS=100
STEPS_EP=200

run_one() {
    local agent_idx=$1
    local seed=$2
    local logfile="$LOGDIR/resume_agent${agent_idx}_seed${seed}.log"
    echo "[$(date '+%H:%M:%S')] Starting agent=$agent_idx seed=$seed  →  $logfile"
    "$RUNNER" "$SCRIPT" \
        --mock \
        --agent_idx   "$agent_idx" \
        --seeds       1 \
        --seed_offset "$seed" \
        --ckpt_suffix "_full_seed${seed}_agent${agent_idx}" \
        --episodes    "$EPISODES" \
        --task_episodes "$TASK_EPS" \
        --steps_per_ep  "$STEPS_EP" \
        2>&1 | tee "$logfile"
    echo "[$(date '+%H:%M:%S')] Done    agent=$agent_idx seed=$seed"
}

# Run the 6 missing runs sequentially (checkpoint saves after each task switch)
for agent_idx in 3 4; do
    for seed in 0 1 2; do
        run_one "$agent_idx" "$seed"
    done
done

echo ""
echo "=== All resume runs complete ==="
echo "Results in: $REPO/maniskill_vit/results/results/"
