"""
ManiSkill ViT PALR Experiment Runner
======================================
Runs PALR and baseline agents on the 4-task Heterogeneous Skill Stream
(PickCube → StackCube → TurnFaucet → PushCube).

Features:
  - ViT-Small (from scratch) as shared encoder for all agents
  - SAC with PALR, L2Reg, ShrinkPerturb, FixedLR baselines
  - Per-seed checkpoint files (safe to interrupt and resume)
  - Plasticity metrics logged every `log_freq` episodes
  - Video saved every `video_every` episodes

Usage:
    # Full sweep (4 agents × 3 seeds)
    python run_experiments.py

    # Single seed/agent
    python run_experiments.py --seeds 1 --seed_offset 0 --agent_idx 3

    # Fast debug run
    python run_experiments.py --fast
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from mock_env import MockHeterogeneousSkillStream, MOCK_TASK_SEQUENCE
from palr_vit_agent import (
    AGENT_REGISTRY, AGENT_NAMES,
    PALRViTAgent, PALRNoPerturb, PALRNoScale,
)
from sac_agent import SACAgent, SACL2RegAgent, SACShrinkPerturbAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------
def to_serialisable(obj):
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, dict):            return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):            return [to_serialisable(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def make_agent(
    agent_idx:       int,
    obs_shape:       tuple,
    action_dim:      int,
    device:          str,
    lr:              float,
    buffer_capacity: int = 50_000,
    batch_size:      int = 256,
) -> SACAgent:
    """Instantiate the correct agent class."""
    cls_map = {
        0: (SACAgent,            {}),
        1: (SACL2RegAgent,       {"l2_coef": 1e-4}),
        2: (SACShrinkPerturbAgent, {"perturb_freq": 1000, "shrink_factor": 0.9, "perturb_std": 0.01}),
        3: (PALRViTAgent,        {"beta_dead": 0.0, "beta_rank": 1.5, "sigma": 0.05,
                                   "dead_threshold": 0.01, "palr_freq": 500, "perturb_freq": 500}),
        4: (PALRNoPerturb,       {"beta_dead": 0.0, "beta_rank": 1.5, "sigma": 0.05,
                                   "dead_threshold": 0.01, "palr_freq": 500}),
        5: (PALRNoScale,         {"beta_dead": 0.0, "beta_rank": 1.5, "sigma": 0.05,
                                   "dead_threshold": 0.01, "perturb_freq": 500}),
    }
    cls, extra_kwargs = cls_map[agent_idx]
    return cls(
        obs_shape=obs_shape,
        action_dim=action_dim,
        lr=lr,
        device=device,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        **extra_kwargs,
    )


# ---------------------------------------------------------------------------
# Run one (seed, agent) combination
# ---------------------------------------------------------------------------
def run_one(
    agent_idx:       int,
    seed:            int,
    n_episodes:      int,
    task_episodes:   int,
    steps_per_ep:    int,
    lr:              float,
    device:          str,
    ckpt_suffix:     str,
    log_freq:        int,
    updates_per_step: int,
    warmup_steps:    int,
    use_mock:        bool = False,
    buffer_capacity: int  = 50_000,
    batch_size:      int  = 256,
    obs_size:        int  = 128,
    **kwargs,
) -> dict:
    """
    Train one agent for `n_episodes` and return result dict.
    Saves a checkpoint JSON after each task switch and at end.
    """
    agent_name = AGENT_NAMES[agent_idx]
    ckpt_path  = os.path.join(RESULTS_DIR, f"vit_checkpoint{ckpt_suffix}.json")

    # Resume from checkpoint if it exists
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        if agent_name in ckpt:
            print(f"  [resume] {agent_name} has {len(ckpt[agent_name])} runs, skipping.")
            return ckpt

    print(f"\n{'='*60}")
    print(f"  Agent: {agent_name}  |  seed={seed}  |  episodes={n_episodes}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment (mock = no GPU needed; real = ManiSkill3 requires GPU)
    if use_mock:
        env = MockHeterogeneousSkillStream(
            task_sequence=MOCK_TASK_SEQUENCE,
            task_episodes=task_episodes,
            steps_per_episode=steps_per_ep,
            seed=seed,
            obs_size=obs_size,
        )
    else:
        from maniskill_env import HeterogeneousSkillStream, TASK_SEQUENCE
        env = HeterogeneousSkillStream(
            task_sequence=TASK_SEQUENCE,
            task_episodes=task_episodes,
            steps_per_episode=steps_per_ep,
            seed=seed,
            obs_size=obs_size,
        )
    obs_shape  = env.obs_shape
    action_dim = env.action_dim

    # Agent
    agent = make_agent(agent_idx, obs_shape, action_dim, device, lr,
                       buffer_capacity=buffer_capacity, batch_size=batch_size)
    print(f"  obs_shape={obs_shape}  action_dim={action_dim}  device={device}")

    # Results storage
    episode_rewards:     list = []
    episode_successes:   list = []
    episode_task_ids:    list = []
    task_switch_eps:     list = []
    plasticity_log:      list = []
    palr_history:        list = []

    step_count = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        ep_reward   = 0.0
        ep_success  = 0.0
        ep_steps    = 0

        # Track task switches
        if len(env.task_switch_episodes) > len(task_switch_eps):
            task_switch_eps = env.task_switch_episodes.copy()

        while not done:
            # Random exploration during warmup
            if step_count < warmup_steps:
                action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = env.step(action)

            # Store transition
            agent.buffer.add(obs, action, reward, next_obs, float(done))

            # Update
            for _ in range(updates_per_step):
                agent.update()

            obs          = next_obs
            ep_reward   += reward
            ep_steps    += 1
            step_count  += 1

            if info.get("success", 0):
                ep_success = 1.0

        episode_rewards.append(float(ep_reward))
        episode_successes.append(float(ep_success))
        episode_task_ids.append(int(env.current_task_idx))

        # Plasticity logging
        if ep % log_freq == 0:
            metrics = agent.get_plasticity_metrics()
            metrics["episode"] = ep
            metrics["task"]    = env.current_task_name
            plasticity_log.append(metrics)

            # PALR history
            if hasattr(agent, "palr_history") and agent.palr_history:
                palr_history = agent.palr_history.copy()

            elapsed = time.time() - t0
            mean_r  = np.mean(episode_rewards[-log_freq:]) if episode_rewards else 0.0
            dead_l5 = metrics.get("dead_L5", float("nan"))
            print(
                f"  ep {ep:4d}/{n_episodes} | task={env.current_task_name:<20s} "
                f"| r={mean_r:6.2f} | dead_L5={dead_l5:.3f} | {elapsed:.0f}s"
            )

        # Checkpoint after each task switch episode
        if ep in task_switch_eps or ep == n_episodes - 1:
            result = {
                agent_name: [{
                    "seed":              seed,
                    "episode_rewards":   episode_rewards,
                    "episode_successes": episode_successes,
                    "episode_task_ids":  episode_task_ids,
                    "task_switch_episodes": task_switch_eps,
                    "plasticity_log":    plasticity_log,
                    "palr_history":      palr_history,
                    "agent_idx":         agent_idx,
                }]
            }
            with open(ckpt_path, "w") as f:
                json.dump(to_serialisable(result), f, indent=2)
            print(f"  [ckpt] saved → {os.path.basename(ckpt_path)}")

    env.close()

    result = {
        agent_name: [{
            "seed":              seed,
            "episode_rewards":   episode_rewards,
            "episode_successes": episode_successes,
            "episode_task_ids":  episode_task_ids,
            "task_switch_episodes": task_switch_eps,
            "plasticity_log":    plasticity_log,
            "palr_history":      palr_history,
            "agent_idx":         agent_idx,
        }]
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("ManiSkill ViT PALR Experiment Runner")
    p.add_argument("--episodes",     type=int, default=400,
                   help="Total episodes per run (default: 400 = 4 tasks × 100 eps)")
    p.add_argument("--task_episodes",type=int, default=100,
                   help="Episodes per task before switching")
    p.add_argument("--steps_per_ep", type=int, default=200,
                   help="Max environment steps per episode")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--seeds",        type=int, default=3,
                   help="Number of seeds to run")
    p.add_argument("--seed_offset",  type=int, default=0,
                   help="Starting seed index")
    p.add_argument("--agent_idx",    type=int, default=None,
                   help="Run single agent index (0-5). Default: all agents.")
    p.add_argument("--ckpt_suffix",  type=str, default="",
                   help="Suffix for checkpoint filename")
    p.add_argument("--log_freq",     type=int, default=10,
                   help="Log plasticity every N episodes")
    p.add_argument("--updates",      type=int, default=1,
                   help="Gradient updates per environment step")
    p.add_argument("--warmup",       type=int, default=1000,
                   help="Random exploration steps before training")
    p.add_argument("--obs_size",      type=int, default=128,
                   help="Observation image size (default: 128; use 32 for fast CPU tests)")
    p.add_argument("--fast",         action="store_true",
                   help="Fast debug run (50 episodes, 1 seed, 32×32 obs)")
    p.add_argument("--mock",         action="store_true",
                   help="Use mock env (no GPU/ManiSkill needed — pipeline test only)")
    return p.parse_args()


def main():
    args = parse_args()

    # --fast: tiny everything so the pipeline test finishes in <60s on CPU
    buffer_cap  = 50_000
    batch_sz    = 256
    obs_sz      = args.obs_size
    if args.fast:
        args.episodes      = 50
        args.seeds         = 1
        args.task_episodes = 12
        args.steps_per_ep  = 20
        args.warmup        = 50
        args.log_freq      = 5
        buffer_cap         = 500    # ≈ 500 × 2 × 768 × 4 B ≈ 3 MB at 32×32
        batch_sz           = 32
        obs_sz             = 32     # 32×32 → 4 patches (vs 64), ~15× faster ViT
        print("[fast mode] 50 eps, 1 seed, 12 eps/task, 20 steps/ep, obs=32×32, buf=500, bs=32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    agent_indices = [args.agent_idx] if args.agent_idx is not None else list(AGENT_NAMES.keys())

    for seed_i in range(args.seeds):
        seed = args.seed_offset + seed_i
        for agent_idx in agent_indices:
            suffix = args.ckpt_suffix or f"_seed{seed}_agent{agent_idx}"
            result = run_one(
                agent_idx        = agent_idx,
                seed             = seed,
                n_episodes       = args.episodes,
                task_episodes    = args.task_episodes,
                steps_per_ep     = args.steps_per_ep,
                lr               = args.lr,
                device           = device,
                ckpt_suffix      = suffix,
                log_freq         = args.log_freq,
                updates_per_step = args.updates,
                warmup_steps     = args.warmup,
                use_mock         = args.mock,
                buffer_capacity  = buffer_cap,
                batch_size       = batch_sz,
                obs_size         = obs_sz,
            )

    print("\n=== All runs complete ===")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
