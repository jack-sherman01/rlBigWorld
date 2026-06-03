#!/bin/bash
#SBATCH --job-name=mv_agent3
#SBATCH --partition=gpua
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=3
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/slurm_agent3_%j.out

#SBATCH --mail-user=heng.zhang@iit.it
#SBATCH --mail-type=END,FAIL

export SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_MUJOCO_GL=egl
export SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_EGL_PLATFORM=surfaceless
export SINGULARITYENV_SAPIEN_HEADLESS=1
export SINGULARITYENV_TF_FORCE_GPU_ALLOW_GROWTH=true
export SINGULARITYENV_WANDB_API_KEY=${WANDB_API_KEY}

container_path=/work/hezhang/rlBigWorld/maniskill_vit.sif

module load intel/singularity/singularity-4.2.2
source ~/.wandb_secrets 2>/dev/null || true

WORKDIR=/work/hezhang/rlBigWorld
echo "[$(date)] WORKDIR=${WORKDIR}  agent=3 (PALR-SAC)"

run_one() {
    local seed=$1 gpu=$2
    echo "[$(date)] START agent=3 seed=${seed} gpu=${gpu}"
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=${gpu} \
    singularity exec --disable-cache --nv \
        -B ${WORKDIR}:/workspace \
        ${container_path} \
        bash -c "
            cd /workspace
            unset DISPLAY
            python maniskill_vit/src/run_experiments.py \
                --seeds 1 \
                --seed_offset ${seed} \
                --agent_idx 3 \
                --episodes ${EPISODES:-400} \
                --task_episodes ${TASK_EPS:-100} \
                --ckpt_suffix _full_seed${seed}_agent3 \
                --wandb \
                --wandb_project rlBigWorld-maniskill
        " > logs/agent3_seed${seed}.log 2>&1
    echo "[$(date)] DONE  agent=3 seed=${seed} gpu=${gpu}"
}

run_one 0 0 &
run_one 1 1 &
run_one 2 2 &
wait

echo "[$(date)] All seeds complete (agent 3 — PALR-SAC)."
