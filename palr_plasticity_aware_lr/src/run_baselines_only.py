"""
Re-run all baseline agents with full data capture (real per-episode rewards
+ plasticity logs). Merges with existing PALR results and overwrites
raw_results.json / summary_stats.json.
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dqn_base import DQNAgent
from baselines import ShrinkAndPerturbAgent, PeriodicResetAgent, L2RegAgent
from train import train_agent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COMMON = dict(
    obs_dim=4, n_actions=2, hidden_sizes=(64, 64),
    buffer_size=20_000, batch_size=32, gamma=0.99,
    target_update_freq=200, epsilon_start=1.0,
    epsilon_end=0.05, epsilon_decay=3_000,
)
N_EPS = 600
EPT   = 150
SEED  = 42


def make_baselines(seed):
    return [
        DQNAgent(lr=1e-3, seed=seed, **COMMON),
        ShrinkAndPerturbAgent(lr=1e-3, seed=seed,
                              perturb_freq=2000, alpha=0.9, sigma=0.01, **COMMON),
        PeriodicResetAgent(lr=1e-3, seed=seed, reset_freq=200, **COMMON),
        L2RegAgent(lr=1e-3, seed=seed, l2_coeff=1e-4, **COMMON),
    ]


def compute_summary(all_results):
    task_switches = [150, 300, 450]
    phase_starts = [0] + task_switches
    phase_ends   = task_switches + [N_EPS]
    summary = {}
    for name, runs in all_results.items():
        rewards = np.array([r["episode_rewards"] for r in runs])
        phase_rewards = {}
        for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
            phase_rewards[f"phase_{i+1}"] = {
                "mean": float(rewards[:, s:e].mean()),
                "std":  float(rewards[:, s:e].std()),
            }
        recovery_speeds = []
        for sw in task_switches:
            for run in runs:
                found = False
                for j in range(sw, min(sw + 100, N_EPS)):
                    if run["episode_rewards"][j] >= 200:
                        recovery_speeds.append(j - sw)
                        found = True
                        break
                if not found:
                    recovery_speeds.append(100)
        summary[name] = {
            "overall_mean_reward": float(rewards.mean()),
            "overall_std_reward":  float(rewards.std()),
            "final_50ep_mean":     float(rewards[:, -50:].mean()),
            "final_50ep_std":      float(rewards[:, -50:].std()),
            "phase_rewards":       phase_rewards,
            "mean_recovery_speed": float(np.mean(recovery_speeds)),
            "std_recovery_speed":  float(np.std(recovery_speeds)),
        }
    return summary


def to_serialisable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, dict): return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_serialisable(i) for i in obj]
    return obj


if __name__ == "__main__":
    # Load existing PALR results
    raw_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(raw_path) as f:
        existing = json.load(f)

    palr_results = {
        k: v for k, v in existing.items() if k.startswith("PALR")
    }

    print(f"Loaded {len(palr_results)} existing PALR result(s). Running baselines...\n")

    baseline_results = {}
    agents = make_baselines(SEED)
    for agent in agents:
        print(f"--- {agent.name} ---")
        result = train_agent(agent, n_episodes=N_EPS, episodes_per_task=EPT,
                             seed=SEED, verbose=True, verbose_every=100)
        baseline_results[agent.name] = [result]
        print(f"  Final 100-ep avg: {np.mean(result['episode_rewards'][-100:]):.1f}")
        n_plast = len(result['plasticity_log'])
        print(f"  Plasticity entries logged: {n_plast}\n")

    all_results = {**baseline_results, **palr_results}

    with open(raw_path, "w") as f:
        json.dump(to_serialisable(all_results), f, indent=2)

    summary = compute_summary(all_results)
    with open(os.path.join(RESULTS_DIR, "summary_stats.json"), "w") as f:
        json.dump(to_serialisable(summary), f, indent=2)

    print("=" * 65)
    print("COMPLETE RESULTS (all real data)")
    print("=" * 65)
    rows = sorted(summary.items(), key=lambda x: x[1]["overall_mean_reward"], reverse=True)
    for name, s in rows:
        plast_entries = len(all_results[name][0]["plasticity_log"])
        print(f"  {name:<28} overall:{s['overall_mean_reward']:>7.1f}  "
              f"final50:{s['final_50ep_mean']:>7.1f}  "
              f"recovery:{s['mean_recovery_speed']:>5.1f}ep  "
              f"plasticity_log:{plast_entries}")
    print("=" * 65)
    print("\nSaved. Run plot_results.py to regenerate figures.")
