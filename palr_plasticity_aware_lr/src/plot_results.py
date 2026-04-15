"""
Plot Experiment Results
=======================
Generates publication-quality figures from saved results:

  Figure 1: Learning curves (smoothed episode reward) across all agents,
            with vertical dashed lines at task switches.

  Figure 2: Per-phase bar chart comparing mean reward across agents.

  Figure 3: Plasticity dynamics -- dead neuron fraction and effective rank
            over training for DQN-FixedLR vs PALR.

  Figure 4: Ablation study bar chart.

All figures saved to plots/.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour scheme ─────────────────────────────────────────────────────────────
COLORS = {
    "DQN-FixedLR":           "#d62728",   # red
    "ShrinkAndPerturb":      "#ff7f0e",   # orange
    "PeriodicReset":         "#9467bd",   # purple
    "L2-Regularisation":     "#8c564b",   # brown
    "PALR (ours)":           "#2ca02c",   # green   <-- our method
    "PALR-NoScale":          "#17becf",   # cyan
    "PALR-NoPerturb":        "#1f77b4",   # blue
}

LINESTYLES = {
    "DQN-FixedLR":           "-",
    "ShrinkAndPerturb":      "--",
    "PeriodicReset":         "-.",
    "L2-Regularisation":     ":",
    "PALR (ours)":           "-",
    "PALR-NoScale":          "--",
    "PALR-NoPerturb":        ":",
}

LINEWIDTHS = {
    "PALR (ours)": 2.5,
}


def smooth(x, window=20):
    if len(x) < window:
        return np.array(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_results():
    with open(os.path.join(RESULTS_DIR, "raw_results.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RESULTS_DIR, "summary_stats.json")) as f:
        summary = json.load(f)
    return raw, summary


# ── Figure 1: Learning Curves ─────────────────────────────────────────────────
def plot_learning_curves(raw, summary, window=20):
    fig, ax = plt.subplots(figsize=(11, 5))

    task_switches = raw[list(raw.keys())[0]][0]["task_switch_episodes"]
    for ts in task_switches:
        ax.axvline(ts, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    if task_switches:
        ax.axvline(task_switches[0], color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.7, label="Task switch")

    for name, runs in raw.items():
        rewards = np.array([r["episode_rewards"] for r in runs])  # (seeds, eps)
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)

        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r, window)
        x = np.arange(len(sm_mean)) + window // 2

        color = COLORS.get(name, "black")
        ls    = LINESTYLES.get(name, "-")
        lw    = LINEWIDTHS.get(name, 1.5)

        ax.plot(x, sm_mean, label=name, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.12, color=color)

    # Task phase labels
    task_names = ["Default", "Heavy-Slow", "Light-Fast", "Default-Heavy"]
    phase_starts = [0] + task_switches
    phase_mids = []
    n_eps = len(raw[list(raw.keys())[0]][0]["episode_rewards"])
    phase_ends = task_switches + [n_eps]
    for s, e in zip(phase_starts, phase_ends):
        phase_mids.append((s + e) / 2)
    for mid, tname in zip(phase_mids, task_names):
        ax.text(mid, 5, tname, ha="center", va="bottom",
                fontsize=7.5, color="gray", style="italic")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.set_title("Continual CartPole: Learning Curves Across Task Phases", fontsize=13)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig1_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Figure 2: Per-phase Bar Chart ─────────────────────────────────────────────
def plot_phase_bars(summary):
    agents = list(summary.keys())
    # Determine number of phases
    sample  = list(summary.values())[0]["phase_rewards"]
    phases  = sorted(sample.keys())
    n_phases = len(phases)
    n_agents = len(agents)

    x = np.arange(n_phases)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(agents):
        means = [summary[name]["phase_rewards"][p]["mean"] for p in phases]
        stds  = [summary[name]["phase_rewards"][p]["std"]  for p in phases]
        offset = (i - n_agents / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=name,
                      color=COLORS.get(name, "gray"), alpha=0.82, capsize=3)

    task_names = ["Default", "Heavy-Slow", "Light-Fast", "Default-Heavy"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {i+1}\n({task_names[i]})"
                        for i in range(n_phases)], fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Per-Phase Performance Comparison", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig2_phase_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Figure 3: Plasticity Dynamics ─────────────────────────────────────────────
def plot_plasticity_dynamics(raw):
    """Compare dead neuron fraction and effective rank for FixedLR vs PALR."""
    agents_to_plot = ["DQN-FixedLR", "PALR (ours)"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for name in agents_to_plot:
        if name not in raw:
            continue
        runs = raw[name]
        # Aggregate plasticity logs across seeds
        all_steps      = []
        all_dead       = []
        all_erank      = []

        for run in runs:
            for entry in run["plasticity_log"]:
                all_steps.append(entry["episode"])
                all_dead.append(entry.get("mean_dead", 0.0))
                all_erank.append(entry.get("mean_erank", 1.0))

        # Average across seeds by binning
        if not all_steps:
            continue
        max_ep = max(all_steps)
        bins   = np.arange(0, max_ep + 10, 10)
        dead_binned  = []
        erank_binned = []
        bin_centers  = []
        for b in range(len(bins) - 1):
            mask = [(bins[b] <= s < bins[b+1]) for s in all_steps]
            if any(mask):
                dead_binned.append(np.mean([all_dead[j] for j, m in enumerate(mask) if m]))
                erank_binned.append(np.mean([all_erank[j] for j, m in enumerate(mask) if m]))
                bin_centers.append((bins[b] + bins[b+1]) / 2)

        color = COLORS.get(name, "black")
        lw    = LINEWIDTHS.get(name, 1.5)
        axes[0].plot(bin_centers, dead_binned, label=name, color=color, linewidth=lw)
        axes[1].plot(bin_centers, erank_binned, label=name, color=color, linewidth=lw)

    # Task switch lines
    sample_run = raw[list(raw.keys())[0]][0]
    for ts in sample_run["task_switch_episodes"]:
        axes[0].axvline(ts, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        axes[1].axvline(ts, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    axes[0].set_ylabel("Dead Neuron Fraction", fontsize=11)
    axes[0].set_title("Plasticity Dynamics: Dead Neurons & Effective Rank", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].set_ylabel("Effective Rank", fontsize=11)
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig3_plasticity_dynamics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Figure 4: Ablation Study ──────────────────────────────────────────────────
def plot_ablation(summary):
    ablation_agents = [
        "DQN-FixedLR",
        "PALR-NoPerturb",
        "PALR-NoScale",
        "PALR (ours)",
    ]
    labels = [
        "DQN-FixedLR\n(baseline)",
        "PALR\n(LR-only)",
        "PALR\n(Perturb-only)",
        "PALR\n(Full, ours)",
    ]
    means  = []
    stds   = []
    colors = []
    for name in ablation_agents:
        if name in summary:
            means.append(summary[name]["final_50ep_mean"])
            stds.append(summary[name]["final_50ep_std"])
            colors.append(COLORS.get(name, "gray"))
        else:
            means.append(0)
            stds.append(0)
            colors.append("gray")

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final 50-Episode Mean Reward", fontsize=12)
    ax.set_title("Ablation Study: PALR Components", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars with values
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{mean:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig4_ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Figure 5: Recovery Speed ──────────────────────────────────────────────────
def plot_recovery_speed(summary):
    agents = [a for a in summary if a in COLORS]
    means  = [summary[a]["mean_recovery_speed"] for a in agents]
    stds   = [summary[a]["std_recovery_speed"]  for a in agents]
    colors = [COLORS.get(a, "gray") for a in agents]

    # Sort by recovery speed (lower is better)
    order = np.argsort(means)
    agents = [agents[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(agents))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=4, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_xlabel("Episodes to Recovery after Task Switch (lower is better)", fontsize=10)
    ax.set_title("Adaptation Speed: Recovery After Task Switches", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig5_recovery_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading results...")
    raw, summary = load_results()

    print("Generating figures...")
    plot_learning_curves(raw, summary, window=20)
    plot_phase_bars(summary)
    plot_plasticity_dynamics(raw)
    plot_ablation(summary)
    plot_recovery_speed(summary)

    print(f"\nAll figures saved to: {PLOTS_DIR}")
