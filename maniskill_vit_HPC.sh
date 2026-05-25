#!/bin/bash
#SBATCH --job-name=maniskill_vit
#SBATCH --partition=gpuv
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mem=28G

#SBATCH --mail-user=heng.zhang@iit.it

export SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_MUJOCO_GL=egl
export SINGULARITYENV_PYOPENGL_PLATFORM=egl
export SINGULARITYENV_EGL_PLATFORM=surfaceless
export SINGULARITYENV_SAPIEN_HEADLESS=1
export SINGULARITYENV_TF_FORCE_GPU_ALLOW_GROWTH=true
export SINGULARITYENV_VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

container_path=/work/hezhang/hrii/rlBigWorld/maniskill_vit.sif

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
            --seed_offset ${SEED:-0} \
            --agent_idx ${AGENT:-3} \
            --episodes ${EPISODES:-400} \
            --task_episodes ${TASK_EPS:-100}
    "
