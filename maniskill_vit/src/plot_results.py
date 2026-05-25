"""
Plot ManiSkill ViT experiment results.

Loads all vit_checkpoint_full_*.json files from results/,
merges by agent name across seeds, and plots:
  1. Per-task reward curves (mean ± std across seeds)
  2. Dead neuron fraction over time (L5)
  3. Effective rank over time (L5)

Usage:
    python maniskill_vit/src/plot_results.py
    python maniskill_vit/src/plot_results.py --suffix fast   # loads vit_checkpoint_fast_*.json
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TASK_NAMES  = ["PickCube-v1", "StackCube-v1", "TurnFaucet-v1", "PushCube-v1"]
AGENT_ORDER = ["SAC-FixedLR", "SAC-L2Reg", "SAC-ShrinkPerturb",
               "PALR-SAC", "PALR-NoPerturb", "PALR-NoScale"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def load_all(suffix="full"):
    """Load all checkpoint files matching the suffix into {agent_name: [run_dict, ...]}."""
    pattern = os.path.join(RESULTS_DIR, f"vit_checkpoint_{suffix}_*.json")
    files   = glob.glob(pattern)
    if not files:
        # fallback: load old-style files (no suffix)
        pattern = os.path.join(RESULTS_DIR, "vit_checkpoint_seed*.json")
        files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {RESULTS_DIR}")

    merged = {}
    for path in sorted(files):
        with open(path) as f:
            data = json.load(f)
        for agent_name, runs in data.items():
            merged.setdefault(agent_name, []).extend(runs)

    print(f"Loaded {len(files)} files.")
    for name, runs in merged.items():
        print(f"  {name}: {len(runs)} seed(s), {len(runs[0]['episode_rewards'])} episodes each")
    return merged


def split_by_task(rewards, task_ids, task_switch_eps, n_tasks=4):
    """Return list of per-task reward arrays."""
    task_ids = np.array(task_ids)
    segments = []
    for t in range(n_tasks):
        mask = task_ids == t
        segments.append(np.array(rewards)[mask])
    return segments


def smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_rewards(merged, suffix):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, task in zip(axes, TASK_NAMES):
        ax.set_title(task.replace("-v1", ""), fontsize=11)
        ax.set_xlabel("Episodes on task")
        ax.set_ylabel("Reward")

    for color, agent_name in zip(COLORS, AGENT_ORDER):
        runs = merged.get(agent_name)
        if not runs:
            continue

        # Split each seed into per-task segments
        per_task = [[] for _ in range(4)]
        for run in runs:
            segments = split_by_task(
                run["episode_rewards"],
                run["episode_task_ids"],
                run["task_switch_episodes"],
            )
            for t, seg in enumerate(segments):
                per_task[t].append(smooth(seg))

        for t, (ax, segs) in enumerate(zip(axes, per_task)):
            if not segs:
                continue
            min_len = min(len(s) for s in segs)
            arr     = np.array([s[:min_len] for s in segs])
            mean    = arr.mean(axis=0)
            std     = arr.std(axis=0)
            xs      = np.arange(len(mean))
            ax.plot(xs, mean, label=agent_name, color=color, linewidth=1.5)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("ManiSkill ViT — Per-task Reward (mean ± std across seeds)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"vit_rewards_{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_plasticity(merged, suffix, metric="dead_L5", ylabel="Dead neuron fraction (L5)"):
    fig, ax = plt.subplots(figsize=(8, 4))

    for color, agent_name in zip(COLORS, AGENT_ORDER):
        runs = merged.get(agent_name)
        if not runs:
            continue

        all_vals = []
        for run in runs:
            eps  = [e["episode"] for e in run["plasticity_log"]]
            vals = [e.get(metric, float("nan")) for e in run["plasticity_log"]]
            all_vals.append(vals)

        min_len = min(len(v) for v in all_vals)
        arr     = np.array([v[:min_len] for v in all_vals], dtype=float)
        mean    = np.nanmean(arr, axis=0)
        std     = np.nanstd(arr, axis=0)
        xs      = eps[:min_len]

        ax.plot(xs, mean, label=agent_name, color=color, linewidth=1.5)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ManiSkill ViT — {ylabel}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f"vit_{metric}_{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


def print_summary(merged):
    print("\n=== Final reward summary (last 20 episodes, mean ± std across seeds) ===")
    for agent_name in AGENT_ORDER:
        runs = merged.get(agent_name)
        if not runs:
            continue
        finals = [np.mean(run["episode_rewards"][-20:]) for run in runs]
        print(f"  {agent_name:<22s}: {np.mean(finals):6.2f} ± {np.std(finals):.2f}  ({len(finals)} seeds)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="full", help="Checkpoint filename suffix (default: full)")
    args = p.parse_args()

    merged = load_all(args.suffix)
    plot_rewards(merged, args.suffix)
    plot_plasticity(merged, args.suffix, "dead_L5", "Dead neuron fraction (L5)")
    plot_plasticity(merged, args.suffix, "erank_L5", "Effective rank (L5)")
    print_summary(merged)


if __name__ == "__main__":
    main()
