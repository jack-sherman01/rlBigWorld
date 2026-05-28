"""
Plot Continual CartPole Learning Curves — workshop paper figure.

Saves: palr_plasticity_aware_lr/plots/fig_cartpole_learning_curves.png

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_cartpole_paper.py
"""

import os, sys, json
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
    "PALR (ours)":        "#1f77b4",   # blue — our method
    "DQN-FixedLR":        "#d62728",   # red
    "L2-Regularisation":  "#8c564b",   # brown
    "ShrinkAndPerturb":   "#ff7f0e",   # orange
    "PeriodicReset":      "#9467bd",   # purple
    "PALR-NoPerturb":     "#7f7f7f",   # gray
    "PALR-NoScale":       "#17becf",   # cyan
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

LINEWIDTHS = {
    "PALR (ours)": 2.5,
}

PHASE_NAMES = ["Default", "Heavy-Slow", "Light-Fast", "Default-Heavy",
               "Default", "Heavy-Slow", "Light-Fast", "Default-Heavy",
               "Default", "Heavy-Slow"]


def smooth(x, w=20):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_checkpoints():
    """Merge per-seed checkpoint files into {agent: [run, ...]} dict."""
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
    if not raw:
        print("No checkpoint files found — falling back to raw_results.json")
        with open(os.path.join(RESULTS_DIR, "raw_results.json")) as f:
            raw = json.load(f)

    fig, ax = plt.subplots(figsize=(13, 5))

    first_key    = list(raw.keys())[0]
    switch_eps   = raw[first_key][0]["task_switch_episodes"]
    n_eps        = len(raw[first_key][0]["episode_rewards"])
    ep_per_phase = n_eps // 10

    # Task-switch vertical lines
    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", lw=0.8, alpha=0.6,
                   label="Task switch" if i == 0 else None)

    # Learning curves
    for name in AGENTS:
        if name not in raw:
            continue
        rewards  = np.array([r["episode_rewards"] for r in raw[name]])
        mean_r   = rewards.mean(axis=0)
        std_r    = rewards.std(axis=0)
        sm_mean  = smooth(mean_r)
        sm_std   = smooth(std_r)
        x        = np.arange(len(sm_mean)) + 10
        color    = COLORS.get(name, "black")
        lw       = LINEWIDTHS.get(name, 1.6)
        ls       = LINESTYLES.get(name, "-")
        ax.plot(x, sm_mean, label=name.replace("PALR", "SPM"), color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.10, color=color)

    # Phase labels just below the top spine
    phase_ends   = switch_eps + [n_eps]
    phase_starts = [0] + switch_eps
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
        mid = (s + e) / 2
        ax.text(mid, 0.97, PHASE_NAMES[i], ha="center", va="top",
                fontsize=6.5, color="#333333", style="italic", fontweight="bold", transform=trans)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=1, bbox_to_anchor=(0, 0.92))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "fig_cartpole_learning_curves_SPM.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot()
