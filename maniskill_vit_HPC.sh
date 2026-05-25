#!/bin/bash
#SBATCH --job-name=maniskill_vit
#SBATCH --partition=gpua
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25            # 2 processes × 4 OMP threads each
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/slurm_%j.out

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
export SINGULARITYENV_VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

container_path=/work/hezhang/rlBigWorld/maniskill_vit.sif

module load intel/singularity/singularity-4.2.2

N_GPUS=2

run_one() {
    local agent=$1 seed=$2 gpu=$3
    echo "[$(date)] START agent=${agent} seed=${seed} gpu=${gpu}"
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=${gpu} \
    singularity exec --disable-cache --nv \
        -B $SLURM_SUBMIT_DIR:/workspace \
        -B /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
        ${container_path} \
        bash -c "
            cd /workspace
            unset DISPLAY
            python maniskill_vit/src/run_experiments.py \
                --seeds 1 \
                --seed_offset ${seed} \
                --agent_idx ${agent} \
                --episodes ${EPISODES:-400} \
                --task_episodes ${TASK_EPS:-100}
        " > logs/agent${agent}_seed${seed}.log 2>&1
    echo "[$(date)] DONE  agent=${agent} seed=${seed} gpu=${gpu}"
}

# Launch all 18 runs (6 agents × 3 seeds), 4 at a time cycling across GPUs
slot=0
pids=()

for agent in 0 1 2 3 4 5; do
    for seed in 0 1 2; do
        gpu=$((slot % N_GPUS))
        run_one $agent $seed $gpu &
        pids+=($!)
        slot=$((slot + 1))

        # Wait for the current batch of N_GPUS to finish before launching more
        if [ $((slot % N_GPUS)) -eq 0 ]; then
            wait "${pids[@]}"
            pids=()
        fi
    done
done

# Wait for any remaining jobs
wait "${pids[@]}"

echo "[$(date)] All runs complete."
