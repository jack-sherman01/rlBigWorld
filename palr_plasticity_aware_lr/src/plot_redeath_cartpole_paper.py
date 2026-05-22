"""
Plot Continual CartPole re-death rate — workshop paper figure.

Saves: palr_plasticity_aware_lr/plots/fig_cartpole_redeath_rate.png

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_redeath_cartpole_paper.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AGENTS = {
    "PALR (ours)":  "#2ca02c",   # green
    "PALR-NoScale": "#17becf",   # cyan
}

RATE_KEYS    = ["redeath_rate_l0", "redeath_rate_l1"]
LAYER_LABELS = ["Layer 1", "Layer 2"]


def load_checkpoints():
    raw = {}
    for i in range(10):
        path = os.path.join(RESULTS_DIR, f"raw_results_checkpoint_seed{i}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            seed_data = json.load(f)
        for agent, runs in seed_data.items():
            raw.setdefault(agent, []).extend(runs)
    return raw


def plot():
    raw = load_checkpoints()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, rkey, llabel in zip(axes, RATE_KEYS, LAYER_LABELS):
        for name, color in AGENTS.items():
            runs = raw.get(name, [])
            all_rates, all_steps = [], []
            for run in runs:
                for h in run.get("palr_plasticity_history", []):
                    r = h.get(rkey, 0.0)
                    all_rates.append(r)
                    all_steps.append(h["step"])

            if not all_steps:
                continue

            max_step  = max(all_steps)
            n_bins    = 30
            edges     = np.linspace(0, max_step, n_bins + 1)
            bin_means, bin_ctrs = [], []
            for b in range(n_bins):
                mask = [(edges[b] <= s < edges[b + 1]) for s in all_steps]
                if any(mask):
                    bin_means.append(np.mean([all_rates[j] for j, m in enumerate(mask) if m]))
                    bin_ctrs.append((edges[b] + edges[b + 1]) / 2)

            ax.plot(bin_ctrs, bin_means, color=color, linewidth=1.8, label=name)
            ax.fill_between(bin_ctrs, 0, bin_means, alpha=0.15, color=color)

        # Task-switch lines (estimate from episode boundaries)
        sample_run = (raw.get("PALR (ours)") or [[]])[0]
        switch_eps = sample_run.get("task_switch_episodes", [])
        hist       = sample_run.get("palr_plasticity_history", [])
        if hist and switch_eps:
            n_eps        = len(sample_run.get("episode_rewards", [])) or 1
            steps_per_ep = hist[-1]["step"] / n_eps
            for sw in switch_eps:
                ax.axvline(sw * steps_per_ep, color="gray",
                           linestyle="--", lw=0.8, alpha=0.6)

        ax.set_title(f"Re-death Rate — {llabel}", fontsize=11)
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel(
        "Re-death Rate\n(fraction of revived neurons dead again)", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cartpole_redeath_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot()
