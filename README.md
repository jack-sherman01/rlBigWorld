# rlBigWorld — PALR: Plasticity-Aware Learning Rates for Continual RL

**PALR** combats catastrophic plasticity loss in continual reinforcement learning by measuring per-layer plasticity (dead-neuron fraction + effective rank) and responding with targeted learning-rate scaling and surgical neuron perturbation. This repository contains experiments across three continual-RL domains:

| Domain | Algorithm | Environment |
|--------|-----------|-------------|
| Continual CartPole | DQN + PALR | 4 physics phases |
| ContinualWorld-10/20 | SAC + PALR | 10–20 MetaWorld MT10 tasks |
| JellyBeanWorld | DQN + PALR | Visual agent, reward-sign flips |
| Habitat Fetch Rearrangement | DD-PPO + PALR | 4-phase robot curriculum |

---

## Repository Layout

```
rlBigWorld/
├── palr_plasticity_aware_lr/   # CW10/CW20, JBW, CartPole experiments
│   ├── src/
│   │   ├── palr_agent.py           # PALR-DQN agent
│   │   ├── palr_sac_agent.py       # PALR-SAC agent (CW experiments)
│   │   ├── sac_base.py             # Base SAC (actor + twin critics + auto-α)
│   │   ├── dqn_base.py             # Base DQN
│   │   ├── cw_env.py               # ContinualWorld wrapper (10/20 tasks)
│   │   ├── jbw_env.py              # JellyBeanWorld wrapper
│   │   ├── cw_baselines.py         # SAC baselines (L2Reg, S&P, PeriodicReset)
│   │   ├── baselines.py            # DQN baselines
│   │   ├── plasticity_metrics.py   # dead_neuron_fraction, effective_rank, etc.
│   │   ├── replay_buffer.py        # Continuous replay buffer
│   │   ├── run_cw_experiments.py   # Main CW10/CW20 launcher
│   │   ├── run_jbw_experiments.py  # JBW launcher
│   │   ├── run_experiments.py      # CartPole launcher
│   │   ├── merge_cw_checkpoints.py # Aggregate per-seed results → JSON
│   │   └── plot_*.py               # Plotting scripts
│   └── setup_jbw.sh                # JBW install + smoke test
│
├── palr_habitat/               # Vision-based Fetch Rearrangement (DD-PPO)
│   ├── src/
│   │   ├── palr_trainer.py         # DD-PPO training loop (torchrun, multi-GPU)
│   │   ├── palr_fetch_policy.py    # ResNet-18 + GRU actor-critic
│   │   ├── palr_resnet_encoder.py  # PALR hooks on conv layers
│   │   ├── fetch_curriculum.py     # 4-phase task curriculum manager
│   │   └── plasticity_metrics_cnn.py  # dead_filter_fraction for conv layers
│   └── configs/
│       ├── ddppo_palr_fetch.yaml       # Full PALR config
│       ├── ddppo_baseline_fetch.yaml   # Baseline (no PALR)
│       ├── ddppo_palr_lr_only_fetch.yaml    # Ablation: LR scaling only
│       ├── ddppo_palr_perturb_only_fetch.yaml # Ablation: perturbation only
│       └── rearrange_base.yaml         # Sensors, task registration
│
├── palr_container.def          # Singularity/Apptainer build definition
├── palr_container.sif          # Pre-built image (6.8 GB)
├── run_cw20.sh                 # Production CW20 launcher (Singularity)
├── run.sh                      # Simple local launcher (.venv + GPU paths)
├── run_interactive.sh          # Habitat single-seed training
├── run_interactive_ablation.sh # Habitat ablation sweep
└── run_on_HPC.sh               # SLURM + Singularity HPC launcher
```

---

## Container

The Singularity image packages two conda environments to work around a hard dependency conflict: `habitat-sim 0.3.2` requires Python 3.9, while `mujoco ≥ 3.0.0` (required by MetaWorld 3.0.0) ships Python 3.10+ wheels only.

| Conda env | Python | Key packages | Used for |
|-----------|--------|--------------|----------|
| `palr_plasticity` | **3.10** (default) | PyTorch 2.3.1+cu121, mujoco 3.7.0, metaworld 3.0.0, gymnasium, tensorflow-cpu 2.15.0, jbw | CW10/CW20, JBW, CartPole |
| `palr_habitat` | **3.9** | PyTorch 2.3.1+cu121, habitat-sim 0.3.2 (EGL+Bullet), habitat-lab/baselines 0.3.220241205 | Fetch Rearrangement |

### Build

```bash
# Rootless (preferred — no sudo required):
apptainer build palr_container.sif palr_container.def

# With fakeroot (Singularity ≥ 3.5):
singularity build --fakeroot palr_container.sif palr_container.def
```

Build takes ~30–60 minutes. No GPU required on the build host.

### Runtime — switching environments

The container's default `PATH` points to `palr_plasticity` (Python 3.10). To use the habitat env, invoke binaries by full path:

```bash
# Default (palr_plasticity):
singularity exec --nv palr_container.sif python script.py

# Habitat env (palr_habitat):
singularity exec --nv palr_container.sif \
    /opt/miniforge3/envs/palr_habitat/bin/python script.py
```

---

## Quick Start

### ContinualWorld-20 (recommended entry point)

```bash
# Full sweep — 5 agents × 3 seeds, auto-resumes from checkpoints:
bash run_cw20.sh

# Single agent/seed for debugging:
bash run_cw20.sh --single

# Merge results only (after all runs complete):
bash run_cw20.sh --merge
```

See the [CW20 in Singularity](#running-cw20-experiments-in-singularity) section for full details.

### Running directly (without the launcher)

```bash
# Inside the container, or via run.sh for local .venv:
cd palr_plasticity_aware_lr/src

# CW20 full sweep:
python run_cw_experiments.py --cw20 --episodes_per_task 200 --seeds 3

# Single agent (e.g. PALR, seed 0):
python run_cw_experiments.py --cw20 --agent_idx 4 --seeds 1 --seed_offset 0

# JBW:
python run_jbw_experiments.py --episodes 400 --seeds 3

# CartPole:
python run_experiments.py
```

### Habitat Fetch Rearrangement

```bash
# Multi-GPU training (3 GPUs):
bash run_interactive.sh

# Or manually:
singularity exec --nv \
    --bind /home/hzhang/work/rlBigWorld:/workspace \
    palr_container.sif \
    bash -c "cd /workspace && \
        /opt/miniforge3/envs/palr_habitat/bin/torchrun \
            --nproc_per_node=1 \
            palr_habitat/src/palr_trainer.py \
            --config palr_habitat/configs/ddppo_palr_fetch.yaml \
            --seed 0 --outdir results/palr_seed0"
```

---

## Running CW20 Experiments in Singularity

`run_cw20.sh` is the production launcher. It orchestrates all agents and seeds inside `palr_container.sif`, handles parallelism, logs each run, and auto-resumes from existing checkpoints.

### Prerequisites

```bash
# The .sif must exist before running:
ls palr_container.sif          # 6.8 GB pre-built image
# If missing, build it:
apptainer build palr_container.sif palr_container.def
```

The script auto-detects `apptainer` vs `singularity` and uses whichever is available.

### Modes

```bash
# 1. Full sweep — all agents × all seeds (default: 5 agents × 3 seeds = 15 runs):
bash run_cw20.sh

# 2. Single run — one specific (seed, agent) pair:
SEED=2 AGENT=4 bash run_cw20.sh --single

# 3. Merge only — aggregate existing checkpoints without running anything new:
bash run_cw20.sh --merge
```

### Environment Variable Reference

All knobs can be overridden by prefixing the command, e.g. `EPISODES=400 SEEDS="0 1" bash run_cw20.sh`.

| Variable | Default | Description |
|----------|---------|-------------|
| `SIF` | `<proj_dir>/palr_container.sif` | Path to the Singularity image |
| `PROJ_DIR` | directory of `run_cw20.sh` | Project root (auto-detected) |
| `EPISODES` | `200` | Episodes per task (×20 tasks = 4 000 total per agent) |
| `BATCH_SIZE` | `2048` | SAC replay-buffer batch size |
| `SEEDS` | `"0 1 2"` | Space-separated seed indices to run |
| `AGENTS` | `"0 3 4 5 6"` | Space-separated agent indices to run |
| `PARALLEL` | `5` | How many runs to launch concurrently before waiting |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU to use (passed inside the container) |

### Agent Index Reference

| Index | Agent | Description |
|-------|-------|-------------|
| `0` | SAC-FixedLR | Vanilla SAC, no adaptation — primary baseline |
| `3` | SAC-L2Reg | L2 weight regularisation on the critic loss |
| `4` | **PALR-SAC (ours)** | Per-layer LR scaling + targeted neuron perturbation |
| `5` | PALR-NoPerturb | Ablation: LR scaling only |
| `6` | PALR-NoScale | Ablation: perturbation only |

> Indices `1` (ShrinkAndPerturb) and `2` (PeriodicReset) exist in the code but are excluded from the default `AGENTS` sweep.

### What Runs Inside the Container

Each `launch_one seed agent` call executes:

```bash
apptainer exec --nv \
    --bind <PROJ_DIR>:/workspace \
    palr_container.sif \
    bash -c "
        cd /workspace
        export CUDA_VISIBLE_DEVICES=0
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export EGL_PLATFORM=surfaceless
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export OMP_NUM_THREADS=4
        unset DISPLAY
        python palr_plasticity_aware_lr/src/run_cw_experiments.py \
            --cw20 \
            --episodes_per_task 200 \
            --seeds 1 \
            --batch_size 2048 \
            --seed_offset <seed> \
            --agent_idx <agent> \
            --ckpt_suffix _cw20_seed<seed>_agent<agent>
    "
```

`--nv` injects the host NVIDIA driver and EGL ICD into the container. The project root is bind-mounted at `/workspace` so results written there persist on the host.

### Parallelism and Scheduling

The launcher batches runs in groups of `PARALLEL` (default 5 — one batch per seed, all agents in parallel). After each batch it polls `pgrep -f run_cw_experiments.py` every 60 seconds and waits for all processes to exit before starting the next batch.

```
seed=0: agents [0, 3, 4, 5, 6] launched in parallel ──► wait ──►
seed=1: agents [0, 3, 4, 5, 6] launched in parallel ──► wait ──►
seed=2: agents [0, 3, 4, 5, 6] launched in parallel ──► wait ──►
merge checkpoints
```

To run seeds in parallel across multiple GPUs, launch the script multiple times with different `SEEDS` and `CUDA_VISIBLE_DEVICES`:

```bash
SEEDS="0"   CUDA_VISIBLE_DEVICES=0 bash run_cw20.sh &
SEEDS="1"   CUDA_VISIBLE_DEVICES=1 bash run_cw20.sh &
SEEDS="2"   CUDA_VISIBLE_DEVICES=2 bash run_cw20.sh &
wait
bash run_cw20.sh --merge
```

### Resume Behaviour

`run_cw_experiments.py` auto-resumes from any existing checkpoint:

```
palr_plasticity_aware_lr/results/cw_checkpoint_cw20_seed<S>_agent<A>.json
```

If a checkpoint exists for a given `(seed, agent)` pair, that run is skipped entirely. This means you can safely re-run `bash run_cw20.sh` after a crash or interruption — only missing runs will execute.

### Logs

Each run writes to its own log file:

```
palr_plasticity_aware_lr/results/logs/cw20_seed<S>_agent<A>.log
```

To tail a running experiment:
```bash
tail -f palr_plasticity_aware_lr/results/logs/cw20_seed0_agent4.log
```

### Post-Run: Merge and Plot

After all runs complete, the launcher auto-merges checkpoints. You can also do this manually:

```bash
# Merge all cw20 checkpoints → cw20_raw_results.json:
bash run_cw20.sh --merge

# Or directly inside the container:
apptainer exec --nv --bind $(pwd):/workspace palr_container.sif \
    bash -c "cd /workspace && python \
        palr_plasticity_aware_lr/src/merge_cw_checkpoints.py --suffix cw20"

# Generate plots:
apptainer exec --nv --bind $(pwd):/workspace palr_container.sif \
    bash -c "cd /workspace/palr_plasticity_aware_lr/src && \
        python plot_cw_results.py"
```

Outputs:
```
palr_plasticity_aware_lr/results/cw20_raw_results.json   # merged data
palr_plasticity_aware_lr/plots/                          # figures
```

---

## Algorithms

### PALR — Plasticity-Aware Learning Rates

Every `measure_freq` training steps, PALR diagnoses each network layer on a held-out diagnostic batch and adapts in two ways:

**1. Per-layer learning-rate scaling**

For each layer `i`, two plasticity metrics are computed:
- `dead_i` — fraction of ReLU units with zero activation (range [0, 1])
- `rank_i` — effective rank of the activation matrix (proxy for feature diversity)

A plasticity deficit is formed and used to scale that layer's gradient:
```
scale_i = 1 + (max_lr_scale − 1) · clip(deficit_i / (1 + rank_beta), 0, 1)
```
Layers with high plasticity loss receive a proportionally larger learning rate.

**2. Targeted neuron perturbation**

When `dead_i > perturb_threshold`, PALR **re-initialises only the dead neurons** from the He initialisation distribution. Healthy neurons are untouched, unlike shrink-and-perturb which perturbs the entire weight matrix.

### Best Hyperparameters (CW10, PALR-SAC)

| Parameter | Value |
|-----------|-------|
| `beta` (LR scale sensitivity) | `0.0` |
| `sigma` (perturbation noise std) | `0.5` |
| `perturb_threshold` | `0.05` |
| `rank_beta` | `1.5` |

### Agent Index Reference

| `--agent_idx` | Agent | Description |
|---------------|-------|-------------|
| 0 | SAC-FixedLR | Vanilla SAC, no adaptation |
| 1 | SAC-L2Reg | L2 weight regularisation on critic loss |
| 2 | SAC-ShrinkAndPerturb | Periodic weight shrink + Gaussian noise |
| 3 | SAC-PeriodicReset | Full network reset every K episodes |
| 4 | PALR (ours) | LR scaling + targeted perturbation |
| 5 | PALR-NoPerturb | Ablation: LR scaling only |
| 6 | PALR-NoScale | Ablation: perturbation only |

---

## Environments

### ContinualWorld-10 / CW20

Wraps MetaWorld MT10: 10 tasks (reach, push, pick-place, door-open, drawer-close, …) executed sequentially. Task switches are **silent** — the agent receives no signal, only a change in reward function.

- State: 39-dimensional
- Actions: 4-dimensional continuous
- CW20: each of the 10 tasks appears twice (shuffled order)

### JellyBeanWorld (JBW)

A 2D grid world with partial observability (11×11 visual patch, 4 discrete actions). Reward sign **flips every `phase_episodes` episodes** without notification — the agent must discover that what was previously good is now bad.

### Habitat Fetch Rearrangement

4-phase curriculum (50M steps each):

| Phase | Task | Description |
|-------|------|-------------|
| 1 | RearrangePick (apple) | Pick up apple from random position |
| 2 | RearrangePick (bowl) | Pick up bowl |
| 3 | RearrangeOpen (fridge) | Open fridge door |
| 4 | RearrangePlace (sink) | Place object in sink |

Observations: RGB + Depth (128×128), joint angles (7), `is_holding` (1), object/goal GPS (6). Uses EGL headless rendering.

---

## Metrics

| Metric | Measures |
|--------|---------|
| **Recovery speed** | Episodes to reach performance threshold after a task switch — primary continual learning metric |
| **Final performance** | Mean reward in the final episode window of each task |
| **Dead neuron fraction** | Fraction of ReLU units with zero activation; lower → more plastic |
| **Effective rank** | Entropy-based rank of activation matrix; higher → more diverse features |

---

## Results Structure

```
palr_plasticity_aware_lr/
├── results/
│   ├── cw_checkpoint_cw20_seed0_agent4.json   # Per-seed, per-agent checkpoints
│   ├── cw_checkpoint_cw20_seed1_agent4.json
│   └── ...
├── cw20_raw_results.json                       # Merged results (after --merge)
└── RESULTS_SUMMARY.txt                         # Human-readable benchmark summary

palr_habitat/
└── results/
    └── palr_seed0/
        ├── checkpoints/                        # Model checkpoints
        ├── tb/                                 # TensorBoard logs
        └── curriculum_events.json              # Task switch timestamps
```

After all runs complete, merge and plot:
```bash
cd palr_plasticity_aware_lr/src
python merge_cw_checkpoints.py --suffix cw20
python plot_cw_results.py
```

---

## Local Development (without container)

```bash
# Activate the .venv (Python 3.10, CUDA 13.0):
./run.sh python palr_plasticity_aware_lr/src/run_cw_experiments.py [args]

# Or manually:
source .venv/bin/activate
export MUJOCO_GL=egl
export TF_FORCE_GPU_ALLOW_GROWTH=true
cd palr_plasticity_aware_lr/src && python run_cw_experiments.py
```

The `.venv` does not include `habitat-sim` (Python 3.9 only). Use the container for Habitat experiments.

---

## HPC / SLURM

```bash
# Submit all seeds in parallel via SLURM:
bash run_on_HPC.sh

# The script builds habitat_v3.sif if not present, then submits
# seeds 0/1/2 in parallel with --sweep flag.
```

---

## Dependencies Summary

| Package | `palr_plasticity` (py3.10) | `palr_habitat` (py3.9) |
|---------|--------------------------|----------------------|
| PyTorch | 2.3.1+cu121 | 2.3.1+cu121 |
| habitat-sim | — | 0.3.2 (EGL + Bullet) |
| habitat-lab | — | 0.3.220241205 |
| mujoco | 3.7.0 | — |
| metaworld | 3.0.0 | — |
| jelly-bean-world | 1.0 | — |
| tensorflow-cpu | 2.15.0 | — |
| gymnasium | 1.3.0 | — |
| gym | 0.26.2 | 0.22.0 |
| wandb | 0.15.12 | 0.15.12 |
| numpy | <2 | <2 |
