"""
Fig JBW3 (new): plasticity dynamics with per-seed variance
==========================================================
Replots fig_jbw3_plasticity as mean +/- s.d. across the 5 seeds in
results/jbw_raw_results.json (read-only). The original figure showed the
pooled mean only, which made it look single-seed.

Saves: plots/fig_jbw3_plasticity_new.png

Usage
-----
    python plot_jbw3_new.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "DQN-FixedLR": "#d62728",
    "PALR (ours)": "#2ca02c",
}
# Secondary encoding on top of red/green (colorblind-safe): DQN dashed, PALR solid
LINESTYLES = {
    "DQN-FixedLR": "--",
    "PALR (ours)": "-",
}
BIN_WIDTH = 5  # episodes per bin, same as the original figure


def bin_run(log, bins, key, default):
    """Bin one seed's plasticity_log into per-bin means; NaN where empty."""
    steps = np.array([e["episode"] for e in log])
    vals  = np.array([e.get(key, default) for e in log])
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        mask = (bins[b] <= steps) & (steps < bins[b + 1])
        if mask.any():
            out[b] = vals[mask].mean()
    return out


def main():
    with open(os.path.join(RESULTS_DIR, "jbw_raw_results.json")) as f:
        raw = json.load(f)

    agents_to_plot = ["DQN-FixedLR", "PALR (ours)"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    max_ep = max(e["episode"]
                 for name in agents_to_plot
                 for run in raw[name]
                 for e in run["plasticity_log"])
    bins    = np.arange(0, max_ep + BIN_WIDTH, BIN_WIDTH)
    centers = (bins[:-1] + bins[1:]) / 2

    n_seeds = None
    for name in agents_to_plot:
        runs = raw[name]
        n_seeds = len(runs)
        dead  = np.array([bin_run(r["plasticity_log"], bins, "mean_dead",  0.0)
                          for r in runs])
        erank = np.array([bin_run(r["plasticity_log"], bins, "mean_erank", 1.0)
                          for r in runs])

        color = COLORS[name]
        ls    = LINESTYLES[name]
        lw    = 2.5 if name == "PALR (ours)" else 1.5
        for ax, arr in zip(axes, (dead, erank)):
            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr, axis=0)
            valid = ~np.isnan(mean)
            ax.plot(centers[valid], mean[valid], label=name,
                    color=color, linestyle=ls, linewidth=lw)
            ax.fill_between(centers[valid],
                            (mean - std)[valid], (mean + std)[valid],
                            color=color, alpha=0.15, linewidth=0)

    switch_eps = raw[agents_to_plot[0]][0]["task_switch_episodes"]
    for ax in axes:
        for sw in switch_eps:
            ax.axvline(sw, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("Dead Neuron Fraction ↓", fontsize=11)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"JBW Plasticity Dynamics: PALR vs DQN "
                      f"({n_seeds} seeds, mean ± s.d.)", fontsize=13)
    axes[1].set_ylabel("Effective Rank ↑", fontsize=11)
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw3_plasticity_new.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
