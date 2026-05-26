#!/bin/bash
#SBATCH --job-name=mv_array
#SBATCH --partition=gpua
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --array=0-17
#SBATCH --output=logs/slurm_%A_%a.out

#SBATCH --mail-user=heng.zhang@iit.it
#SBATCH --mail-type=END,FAIL

# Array index → agent and seed mapping (6 agents × 3 seeds = 18 tasks)
# task 0-2: agent 0 seeds 0-2
# task 3-5: agent 1 seeds 0-2  ...
AGENT=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))

echo "[$(date)] array_task=${SLURM_ARRAY_TASK_ID}  agent=${AGENT}  seed=${SEED}"

export SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_MUJOCO_GL=egl
export SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_EGL_PLATFORM=surfaceless
export SINGULARITYENV_SAPIEN_HEADLESS=1
export SINGULARITYENV_TF_FORCE_GPU_ALLOW_GROWTH=true

container_path=/work/hezhang/rlBigWorld/maniskill_vit.sif

module load intel/singularity/singularity-4.2.2

WORKDIR=/work/hezhang/rlBigWorld

# Conditionally bind Vulkan ICD only if the directory exists on this node
VULKAN_BIND=""
for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if [[ -d "${icd_dir}" ]]; then
        VULKAN_BIND="-B ${icd_dir}:${icd_dir}:ro"
        break
    fi
done

singularity exec --disable-cache --nv \
    -B ${WORKDIR}:/workspace \
    ${VULKAN_BIND} \
    ${container_path} \
    bash -c "
        cd /workspace
        unset DISPLAY
        export CUDA_VISIBLE_DEVICES=0
        python maniskill_vit/src/run_experiments.py \
            --seeds 1 \
            --seed_offset ${SEED} \
            --agent_idx ${AGENT} \
            --episodes ${EPISODES:-400} \
            --task_episodes ${TASK_EPS:-100} \
            --ckpt_suffix _full_seed${SEED}_agent${AGENT}
    " > logs/agent${AGENT}_seed${SEED}.log 2>&1

echo "[$(date)] DONE  agent=${AGENT}  seed=${SEED}"
