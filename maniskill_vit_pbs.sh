#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=28gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N maniskill_vit
#PBS -q gpu

#PBS -M heng.zhang@iit.it
#PBS -m b
#PBS -m e

container_path=/home/hzhang/work/rlBigWorld/maniskill_vit.sif

cd $PBS_O_WORKDIR

VULKAN_BIND=""
VK_ICD_ENV=""
for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if [[ -d "${icd_dir}" ]]; then
        VULKAN_BIND="-B ${icd_dir}:${icd_dir}:ro"
        VK_ICD_ENV="export VK_ICD_FILENAMES=${icd_dir}/nvidia_icd.json"
        break
    fi
done

singularity exec --disable-cache --nv \
    -B $PBS_O_WORKDIR:/workspace \
    ${VULKAN_BIND} \
    ${container_path} \
    bash -c "
        cd /workspace
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export EGL_PLATFORM=surfaceless
        export SAPIEN_HEADLESS=1
        ${VK_ICD_ENV}
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export OMP_NUM_THREADS=4
        unset DISPLAY
        python maniskill_vit/src/run_experiments.py \
            --seeds 1 \
            --seed_offset ${SEED:-0} \
            --agent_idx ${AGENT:-3} \
            --episodes ${EPISODES:-400} \
            --task_episodes ${TASK_EPS:-100}
    "
