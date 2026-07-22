"""
CBP Baseline Comparison Figure
================================
Generates fig_cbp_comparison.png — a side-by-side comparison of
CBP-SAC vs SAC-FixedLR vs PALR-SAC (ours) on CW10.

Left panel:  smoothed learning curves with task-switch markers
Right panel: per-task mean reward bar chart

Does NOT overwrite any existing figures.

Usage
-----
    bash run.sh palr_plasticity_aware_lr/src/plot_cbp_comparison.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from cw_env import CW10_TASKS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AGENTS = ["SAC-FixedLR", "CBP-SAC", "PALR-SAC (ours)"]

COLORS = {
    "SAC-FixedLR":     "#d62728",
    "CBP-SAC":         "#e377c2",
    "PALR-SAC (ours)": "#2ca02c",
}
LINESTYLES = {
    "SAC-FixedLR":     "--",
    "CBP-SAC":         "-.",
    "PALR-SAC (ours)": "-",
}
TASK_SHORT = [
    "Reach", "Push", "Pick\nPlace", "Door\nOpen", "Drawer\nOpen",
    "Drawer\nClose", "Button\nPress", "Peg\nInsert", "Window\nOpen", "Window\nClose",
]


def smooth(x, w=20):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load():
    with open(os.path.join(RESULTS_DIR, "cw_raw_results.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RESULTS_DIR, "cw_summary_stats.json")) as f:
        summary = json.load(f)
    return raw, summary


def plot_cbp_comparison(raw, summary, window=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: Learning curves ─────────────────────────────────────────────────
    ax = axes[0]
    switch_eps = raw[list(raw.keys())[0]][0]["task_switch_episodes"]
    n_eps      = len(raw[list(raw.keys())[0]][0]["episode_rewards"])
    ep_per_task = n_eps // len(CW10_TASKS)

    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", lw=0.8, alpha=0.5,
                   label="Task switch" if i == 0 else None)

    for i, tname in enumerate(TASK_SHORT):
        s = i * ep_per_task
        e = (i + 1) * ep_per_task
        ax.text((s + e) / 2, -50, tname, ha="center", va="top",
                fontsize=6, color="gray", style="italic")

    for name in AGENTS:
        if name not in raw:
            print(f"  WARNING: {name} not in raw results, skipping")
            continue
        runs    = raw[name]
        rewards = np.array([r["episode_rewards"] for r in runs])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        x       = np.arange(len(sm_mean)) + window // 2
        color   = COLORS[name]
        lw      = 2.5 if "ours" in name else 1.8
        ax.plot(x, sm_mean, label=f"{name} (n={len(runs)})",
                color=color, linestyle=LINESTYLES[name], linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.12, color=color)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=11)
    ax.set_title("CW10: Learning Curves — CBP vs Baselines", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: Per-task bar chart ─────────────────────────────────────────────
    ax = axes[1]
    n_tasks  = len(CW10_TASKS)
    n_agents = len(AGENTS)
    x        = np.arange(n_tasks)
    width    = 0.7 / n_agents

    for i, name in enumerate(AGENTS):
        if name not in summary:
            continue
        means = [summary[name]["task_means"].get(f"task_{t+1}", 0.0)
                 for t in range(n_tasks)]
        stds  = [summary[name].get("task_stds", {}).get(f"task_{t+1}", 0.0)
                 for t in range(n_tasks)]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=name,
               color=COLORS[name], alpha=0.85,
               yerr=stds, capsize=3, error_kw={"linewidth": 1})

    # Annotate overall means
    y_max = ax.get_ylim()[1]
    for i, name in enumerate(AGENTS):
        if name not in summary:
            continue
        overall = summary[name].get("overall_mean", 0.0)
        offset  = (i - n_agents / 2 + 0.5) * width
        ax.text(n_tasks - 1 + offset, overall + 20,
                f"{overall:.0f}", ha="center", fontsize=7,
                color=COLORS[name], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT, fontsize=8)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title("CW10: Per-Task Performance — CBP vs Baselines", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cbp_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading CW10 results...")
    raw, summary = load()
    print(f"Agents in data: {list(raw.keys())}")
    print("Generating CBP comparison figure...")
    plot_cbp_comparison(raw, summary)
    print("Done.")
