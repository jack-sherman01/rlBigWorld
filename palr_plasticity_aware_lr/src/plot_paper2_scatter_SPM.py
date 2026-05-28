"""
Reproduce fig_paper2_scatter.png with:
  - No title
  - PALR → SPM (suffixes preserved)
Saves: palr_plasticity_aware_lr/plots/fig_paper2_scatter_SPM.png

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_paper2_scatter_SPM.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# internal key → display label
DISPLAY = {
    "PALR-SAC (ours)":    "SPM (ours)",
    "PALR-SAC-NoScale":   "SPM-NoScale",
    "PALR-SAC-NoPerturb": "SPM-NoPerturb",
    "SAC-FixedLR":        "SAC-FixedLR",
    "SAC-L2Reg":          "SAC-L2Reg",
}

COLORS = {
    "PALR-SAC (ours)":    "#1f77b4",
    "PALR-SAC-NoScale":   "#ff7f0e",
    "PALR-SAC-NoPerturb": "#d62728",
    "SAC-FixedLR":        "#7f7f7f",
    "SAC-L2Reg":          "#2ca02c",
}


def plot():
    with open(os.path.join(RESULTS_DIR, "cw_raw_results.json")) as f:
        raw = json.load(f)

    agents = list(DISPLAY.keys())

    means_dead, stds_dead = [], []
    means_rew,  stds_rew  = [], []

    for name in agents:
        runs = raw.get(name, [])
        per_run_dead, per_run_rew = [], []
        for run in runs:
            dead_vals = [e["layer_3_dead"] for e in run["plasticity_log"]]
            per_run_dead.append(np.mean(dead_vals))
            per_run_rew.append(np.mean(run["episode_rewards"]))
        means_dead.append(np.mean(per_run_dead))
        stds_dead.append(np.std(per_run_dead))
        means_rew.append(np.mean(per_run_rew))
        stds_rew.append(np.std(per_run_rew))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(agents):
        color  = COLORS[name]
        label  = DISPLAY[name]
        weight = "bold" if name == "PALR-SAC (ours)" else "normal"
        ax.errorbar(
            means_dead[i], means_rew[i],
            xerr=stds_dead[i], yerr=stds_rew[i],
            fmt="o", color=color, markersize=12,
            capsize=4, elinewidth=1.4, capthick=1.4,
        )
        ax.text(
            means_dead[i] + 0.002, means_rew[i] + 8,
            label, color=color, fontsize=10, fontweight=weight,
        )

    # trend line
    r, p = stats.pearsonr(means_dead, means_rew)
    x_line = np.linspace(min(means_dead) - 0.02, max(means_dead) + 0.02, 100)
    slope, intercept, *_ = stats.linregress(means_dead, means_rew)
    ax.plot(x_line, slope * x_line + intercept,
            color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.97, 0.97, f"Pearson r = {r:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel("Mean Layer-4 Dead-Neuron Fraction  ↓", fontsize=12)
    ax.set_ylabel("Mean Episode Reward  ↑", fontsize=12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig_paper2_scatter_SPM.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot()
