"""
Merge per-seed checkpoint files into a single raw_results.json.

Usage:
    python merge_checkpoints.py [--n_seeds 10] [--episodes 2000]

Expects files:
    results/raw_results_checkpoint_seed0.json
    results/raw_results_checkpoint_seed1.json
    ...
    results/raw_results_checkpoint_seed<N-1>.json

Writes:
    results/raw_results.json
    results/summary_stats.json
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--episodes_per_task", type=int, default=200)
    args = parser.parse_args()

    merged = {}
    found = 0
    missing = []

    for i in range(args.n_seeds):
        fname = os.path.join(RESULTS_DIR, f"raw_results_checkpoint_seed{i}.json")
        if not os.path.exists(fname):
            missing.append(i)
            print(f"  [MISSING] seed {i}: {fname}")
            continue
        with open(fname) as f:
            seed_data = json.load(f)
        for agent_name, runs in seed_data.items():
            if agent_name not in merged:
                merged[agent_name] = []
            merged[agent_name].extend(runs)
        found += 1
        print(f"  [OK] seed {i}: {sum(len(v) for v in seed_data.values())} runs")

    if not merged:
        print("No checkpoint files found. Exiting.")
        sys.exit(1)

    print(f"\nMerged {found}/{args.n_seeds} seeds. "
          f"Total runs per agent: {[len(v) for v in merged.values()]}")
    if missing:
        print(f"WARNING: missing seeds: {missing}")

    # Save merged raw results
    raw_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged raw results → {raw_path}")

    # Compute summary stats
    n_episodes = args.episodes
    summary = {}
    for name, runs in merged.items():
        all_rewards = np.array([r["episode_rewards"] for r in runs])
        switch_ep   = runs[0]["task_switch_episodes"]

        phase_rewards = {}
        phase_starts = [0] + switch_ep
        phase_ends   = switch_ep + [n_episodes]
        for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
            phase_rewards[f"phase_{i+1}"] = {
                "mean": float(all_rewards[:, s:e].mean()),
                "std":  float(all_rewards[:, s:e].std()),
            }

        recovery_speeds = []
        for sw in switch_ep:
            for run in runs:
                found_r = False
                for j in range(sw, min(sw + 100, n_episodes)):
                    if run["episode_rewards"][j] >= 200:
                        recovery_speeds.append(j - sw)
                        found_r = True
                        break
                if not found_r:
                    recovery_speeds.append(100)

        summary[name] = {
            "n_runs":                int(len(runs)),
            "overall_mean_reward":   float(all_rewards.mean()),
            "overall_std_reward":    float(all_rewards.std()),
            "final_50ep_mean":       float(all_rewards[:, -50:].mean()),
            "final_50ep_std":        float(all_rewards[:, -50:].std()),
            "phase_rewards":         phase_rewards,
            "mean_recovery_speed":   float(np.mean(recovery_speeds)),
            "std_recovery_speed":    float(np.std(recovery_speeds)),
        }

    summary_path = os.path.join(RESULTS_DIR, "summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary stats → {summary_path}")

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    header = f"{'Agent':<28} {'N':>4} {'Overall Mean':>14} {'Final 50ep':>12} {'Recovery':>12}"
    print(header)
    print("-"*80)
    rows = sorted(summary.items(), key=lambda x: x[1]["overall_mean_reward"], reverse=True)
    for name, stats in rows:
        print(
            f"{name:<28} "
            f"{stats['n_runs']:>4} "
            f"{stats['overall_mean_reward']:>10.1f}±{stats['overall_std_reward']:<5.1f} "
            f"{stats['final_50ep_mean']:>8.1f}±{stats['final_50ep_std']:<5.1f} "
            f"{stats['mean_recovery_speed']:>8.1f}ep"
        )
    print("="*80)


if __name__ == "__main__":
    main()
