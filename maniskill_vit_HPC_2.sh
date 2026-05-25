#!/bin/bash
#SBATCH --job-name=mv_agents23
#SBATCH --partition=gpua
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/slurm_agents23_%j.out

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

container_path=/work/hezhang/rlBigWorld/maniskill_vit.sif

module load intel/singularity/singularity-4.2.2

N_GPUS=4
WORKDIR=/work/hezhang/rlBigWorld
echo "[$(date)] WORKDIR=${WORKDIR}  agents=2,3"

run_one() {
    local agent=$1 seed=$2 gpu=$3
    echo "[$(date)] START agent=${agent} seed=${seed} gpu=${gpu}"
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
                --agent_idx ${agent} \
                --episodes ${EPISODES:-400} \
                --task_episodes ${TASK_EPS:-100} \
                --ckpt_suffix _full_seed${seed}_agent${agent}
        " > logs/agent${agent}_seed${seed}.log 2>&1
    echo "[$(date)] DONE  agent=${agent} seed=${seed} gpu=${gpu}"
}

slot=0
pids=()

for agent in 2 3; do
    for seed in 0 1 2; do
        gpu=$((slot % N_GPUS))
        run_one $agent $seed $gpu &
        pids+=($!)
        slot=$((slot + 1))
        if [ $((slot % N_GPUS)) -eq 0 ]; then
            wait "${pids[@]}"
            pids=()
        fi
    done
done

wait "${pids[@]}"
echo "[$(date)] All runs complete (agents 2,3)."
