#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=28gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N ms_resume
#PBS -q gpu

#PBS -M heng.zhang@iit.it
#PBS -m b
#PBS -m e

# Resume PALR-SAC or PALR-NoPerturb from existing checkpoint.
# Pass SEED and AGENT as qsub -v variables, e.g.:
#   qsub -v SEED=0,AGENT=3 maniskill_vit_resume_pbs.sh

container_path=/home/hzhang/work/rlBigWorld/maniskill_vit.sif

cd $PBS_O_WORKDIR

SEED="${SEED:-0}"
AGENT="${AGENT:-3}"

echo "[resume] seed=${SEED} agent=${AGENT} started at $(date)"

singularity exec --disable-cache --nv \
    -B $PBS_O_WORKDIR:/workspace \
    ${container_path} \
    bash -c "
        cd /workspace
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export OMP_NUM_THREADS=4
        export PYTHONUNBUFFERED=1
        python maniskill_vit/src/run_experiments.py \
            --mock \
            --seeds 1 \
            --seed_offset ${SEED} \
            --agent_idx   ${AGENT} \
            --ckpt_suffix _full_seed${SEED}_agent${AGENT} \
            --episodes    400 \
            --task_episodes 100 \
            --steps_per_ep  200
    "

echo "[resume] seed=${SEED} agent=${AGENT} done at $(date)"
