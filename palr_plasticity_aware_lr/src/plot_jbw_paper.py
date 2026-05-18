"""
Plot JellyBeanWorld learning curves — workshop paper figure.

Saves: palr_plasticity_aware_lr/plots/fig_jbw_learning_curves.png

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_jbw_paper.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AGENTS = [
    "PALR (ours)",
    "DQN-FixedLR",
    "L2-Regularisation",
    "ShrinkAndPerturb",
    "PeriodicReset",
    "PALR-NoPerturb",
    "PALR-NoScale",
]

COLORS = {
    "PALR (ours)":        "#1f77b4",
    "DQN-FixedLR":        "#d62728",
    "L2-Regularisation":  "#8c564b",
    "ShrinkAndPerturb":   "#ff7f0e",
    "PeriodicReset":      "#9467bd",
    "PALR-NoPerturb":     "#7f7f7f",
    "PALR-NoScale":       "#17becf",
}

LINESTYLES = {
    "PALR (ours)":        "-",
    "DQN-FixedLR":        "--",
    "L2-Regularisation":  ":",
    "ShrinkAndPerturb":   "-.",
    "PeriodicReset":      "--",
    "PALR-NoPerturb":     "-",
    "PALR-NoScale":       "-",
}

LINEWIDTHS = {"PALR (ours)": 2.5}

PHASE_LABELS = ["Normal", "Inverted", "Normal", "Inverted"]


def smooth(x, w=10):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot():
    with open(os.path.join(RESULTS_DIR, "jbw_raw_results.json")) as f:
        raw = json.load(f)

    fig, ax = plt.subplots(figsize=(13, 5))

    first_key  = list(raw.keys())[0]
    switch_eps = raw[first_key][0]["task_switch_episodes"]
    n_eps      = len(raw[first_key][0]["episode_rewards"])

    # Task-switch vertical lines
    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", lw=0.8, alpha=0.6,
                   label="Reward flip" if i == 0 else None)

    # Learning curves
    for name in AGENTS:
        if name not in raw:
            continue
        rewards = np.array([r["episode_rewards"] for r in raw[name]])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r)
        sm_std  = smooth(std_r)
        x       = np.arange(len(sm_mean)) + 5
        color   = COLORS.get(name, "black")
        lw      = LINEWIDTHS.get(name, 1.6)
        ls      = LINESTYLES.get(name, "-")
        ax.plot(x, sm_mean, label=name, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.10, color=color)

    # Phase labels at top
    from matplotlib.transforms import blended_transform_factory
    trans        = blended_transform_factory(ax.transData, ax.transAxes)
    phase_starts = [0] + switch_eps
    phase_ends   = switch_eps + [n_eps]
    for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
        ax.text((s + e) / 2, 0.97, PHASE_LABELS[i],
                ha="center", va="top", fontsize=7, color="gray",
                style="italic", transform=trans)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig_jbw_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot()
