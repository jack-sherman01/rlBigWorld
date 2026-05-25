#!/bin/bash
#SBATCH --job-name=maniskill_vit
#SBATCH --partition=gpua
#SBATCH --array=0-17                  # 6 agents × 3 seeds = 18 independent runs
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/slurm_%A_%a.out  # %A=job id, %a=array index

#SBATCH --mail-user=heng.zhang@iit.it
#SBATCH --mail-type=END,FAIL

# Map array index → (agent_idx, seed)
# Layout: task 0-2 → agent 0 seeds 0-2, task 3-5 → agent 1 seeds 0-2, ...
AGENT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))

echo "[$(date)] array_task=${SLURM_ARRAY_TASK_ID}  agent=${AGENT_IDX}  seed=${SEED}"

export SINGULARITYENV_OMP_NUM_THREADS=8
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_MUJOCO_GL=egl
export SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_EGL_PLATFORM=surfaceless
export SINGULARITYENV_SAPIEN_HEADLESS=1
export SINGULARITYENV_TF_FORCE_GPU_ALLOW_GROWTH=true
export SINGULARITYENV_VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

container_path=/work/hezhang/rlBigWorld/maniskill_vit.sif

module load intel/singularity/singularity-4.2.2

singularity exec --disable-cache --nv \
    -B $SLURM_SUBMIT_DIR:/workspace \
    -B /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
    ${container_path} \
    bash -c "
        cd /workspace
        unset DISPLAY
        python maniskill_vit/src/run_experiments.py \
            --seeds 1 \
            --seed_offset ${SEED} \
            --agent_idx ${AGENT_IDX} \
            --episodes ${EPISODES:-400} \
            --task_episodes ${TASK_EPS:-100}
    "
