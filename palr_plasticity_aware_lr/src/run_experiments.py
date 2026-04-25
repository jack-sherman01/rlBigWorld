"""
Main Experiment Runner
======================
Runs all agents (baselines + PALR variants) on Continual CartPole,
saves raw results to results/, and triggers plotting.

Usage:
    python run_experiments.py [--episodes 1200] [--seeds 3] [--fast]

    --fast: Run only 400 episodes per seed (for quick debugging)
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dqn_base import DQNAgent
from baselines import ShrinkAndPerturbAgent, PeriodicResetAgent, L2RegAgent
from palr_agent import PALRAgent
from train import train_agent


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Shared hyper-parameters ──────────────────────────────────────────────────
COMMON = dict(
    obs_dim      = 4,
    n_actions    = 2,
    hidden_sizes = (64, 64),
    buffer_size  = 20_000,
    batch_size   = 32,
    gamma        = 0.99,
    target_update_freq = 200,
    epsilon_start = 1.0,
    epsilon_end   = 0.05,
    epsilon_decay = 3_000,
)


def make_agents(seed):
    """Instantiate all agent variants for one seed."""
    return [
        DQNAgent(lr=1e-3, seed=seed, **COMMON),
        ShrinkAndPerturbAgent(lr=1e-3, seed=seed, perturb_freq=2000,
                               alpha=0.9, sigma=0.01, **COMMON),
        PeriodicResetAgent(lr=1e-3, seed=seed, reset_freq=200, **COMMON),
        L2RegAgent(lr=1e-3, seed=seed, l2_coeff=1e-4, **COMMON),
        # PALR full method
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  perturb_sigma=0.3, **COMMON),
        # Ablation: LR scaling only (no perturbation)
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  no_perturb=True, **COMMON),
        # Ablation: perturbation only (no LR scaling)
        PALRAgent(base_lr=1e-3, seed=seed, beta=3.0, rank_beta=1.5,
                  measure_freq=100, perturb_threshold=0.10,
                  perturb_sigma=0.3, no_scale=True, **COMMON),
    ]


def save_checkpoint(all_results, n_episodes, n_seeds, episodes_per_task,
                    ckpt_suffix=""):
    """Save incremental checkpoint after each seed.

    ckpt_suffix: appended before .json so parallel workers don't clobber each
    other (e.g. "_seed0", "_seed3").
    """
    def to_serialisable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_serialisable(i) for i in obj]
        return obj
    fname = f"raw_results_checkpoint{ckpt_suffix}.json"
    ckpt_path = os.path.join(RESULTS_DIR, fname)
    with open(ckpt_path, "w") as f:
        json.dump(to_serialisable(all_results), f)
    print(f"  [checkpoint saved → {ckpt_path}]")


def run_all(n_episodes: int, n_seeds: int, episodes_per_task: int,
            seed_offset_start: int = 0, ckpt_suffix: str = ""):
    all_results = {}  # agent_name -> list of per-seed result dicts

    # Load existing checkpoint if resuming
    fname = f"raw_results_checkpoint{ckpt_suffix}.json"
    ckpt_path = os.path.join(RESULTS_DIR, fname)
    if seed_offset_start > 0 and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            all_results = json.load(f)
        print(f"Loaded checkpoint with {sum(len(v) for v in all_results.values())} runs "
              f"from {ckpt_path}")

    for seed_offset in range(seed_offset_start, seed_offset_start + n_seeds):
        seed = 42 + seed_offset * 13
        print(f"\n{'='*60}")
        print(f"SEED {seed_offset + 1}/{seed_offset_start + n_seeds}  (seed={seed})")
        print(f"{'='*60}")
        agents = make_agents(seed)

        for agent in agents:
            print(f"\n--- {agent.name} ---")
            result = train_agent(
                agent,
                n_episodes=n_episodes,
                episodes_per_task=episodes_per_task,
                seed=seed,
                verbose=True,
                verbose_every=100,
            )
            if agent.name not in all_results:
                all_results[agent.name] = []
            all_results[agent.name].append(result)

        # Save checkpoint after every seed
        save_checkpoint(all_results, n_episodes, n_seeds, episodes_per_task,
                        ckpt_suffix=ckpt_suffix)

    return all_results


def save_results(all_results, n_episodes, n_seeds, episodes_per_task):
    """Save results to JSON for reproducibility."""
    # Convert numpy arrays to lists for JSON serialisation
    def to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serialisable(i) for i in obj]
        return obj

    save_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(save_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)
    print(f"\nSaved raw results to {save_path}")

    # Also save summary stats
    summary = {}
    for name, runs in all_results.items():
        all_rewards = np.array([r["episode_rewards"] for r in runs])
        switch_ep   = runs[0]["task_switch_episodes"]

        # Per-phase mean reward
        phase_rewards = {}
        phase_starts = [0] + switch_ep
        phase_ends   = switch_ep + [n_episodes]
        for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
            phase_rewards[f"phase_{i+1}"] = {
                "mean": float(all_rewards[:, s:e].mean()),
                "std":  float(all_rewards[:, s:e].std()),
            }

        # Recovery speed: episodes to reach 200 reward after each task switch
        recovery_speeds = []
        for switch_ep_i in switch_ep:
            for run in runs:
                found = False
                for j in range(switch_ep_i, min(switch_ep_i + 100, n_episodes)):
                    if run["episode_rewards"][j] >= 200:
                        recovery_speeds.append(j - switch_ep_i)
                        found = True
                        break
                if not found:
                    recovery_speeds.append(100)  # did not recover

        summary[name] = {
            "overall_mean_reward": float(all_rewards.mean()),
            "overall_std_reward":  float(all_rewards.std()),
            "final_50ep_mean":     float(all_rewards[:, -50:].mean()),
            "final_50ep_std":      float(all_rewards[:, -50:].std()),
            "phase_rewards":       phase_rewards,
            "mean_recovery_speed": float(np.mean(recovery_speeds)),
            "std_recovery_speed":  float(np.std(recovery_speeds)),
        }

    summary_path = os.path.join(RESULTS_DIR, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary stats to {summary_path}")
    return summary


def print_summary_table(summary):
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    header = f"{'Agent':<28} {'Overall Mean':>14} {'Final 50ep':>12} {'Recovery':>12}"
    print(header)
    print("-"*80)
    rows = sorted(
        summary.items(),
        key=lambda x: x[1]["overall_mean_reward"],
        reverse=True
    )
    for name, stats in rows:
        print(
            f"{name:<28} "
            f"{stats['overall_mean_reward']:>10.1f}±{stats['overall_std_reward']:<5.1f} "
            f"{stats['final_50ep_mean']:>8.1f}±{stats['final_50ep_std']:<5.1f} "
            f"{stats['mean_recovery_speed']:>8.1f}ep"
        )
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200,
                        help="Total episodes per agent per seed")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds")
    parser.add_argument("--episodes_per_task", type=int, default=200,
                        help="Episodes before task switch")
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: 400 episodes, 1 seed")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Start from this seed index (0-based). Use to resume after interruption.")
    parser.add_argument("--ckpt_suffix", type=str, default="",
                        help="Suffix appended to checkpoint filename (e.g. '_seed0'). "
                             "Allows parallel workers to write separate checkpoint files.")
    args = parser.parse_args()

    if args.fast:
        n_ep    = 600
        n_seeds = 1
        ept     = 150
    else:
        n_ep    = args.episodes
        n_seeds = args.seeds
        ept     = args.episodes_per_task

    print(f"Running: {n_ep} episodes, {n_seeds} seed(s), {ept} ep/task, seed_offset={args.seed_offset}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    all_results = run_all(n_ep, n_seeds, ept, seed_offset_start=args.seed_offset,
                          ckpt_suffix=args.ckpt_suffix)
    summary     = save_results(all_results, n_ep, n_seeds, ept)
    print_summary_table(summary)
    print("\nDone. Run plot_results.py to generate figures.")
