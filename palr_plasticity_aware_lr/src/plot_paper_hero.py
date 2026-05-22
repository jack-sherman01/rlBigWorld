"""
Plot CW10 Hero Figure  (Figure 1 of the PALR paper)
=====================================================
Two-panel figure:
  Panel 1 (top):    Smoothed episode reward across 2000 episodes
  Panel 2 (bottom): Layer-4 dead-neuron fraction

Layout choices (vs the ad-hoc original):
  - No figure title
  - Legend in top-right corner of panel 1
  - Task names at the bottom of panel 2, horizontal

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_paper_hero.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(__file__))
from cw_env import CW10_TASKS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AGENTS = [
    "PALR-SAC (ours)",
    "PALR-SAC-NoScale",
    "SAC-FixedLR",
    "PALR-SAC-NoPerturb",
]

DISPLAY = {
    "PALR-SAC (ours)":    "PALR (ours)",
    "PALR-SAC-NoScale":   "PALR-NoScale",
    "SAC-FixedLR":        "SAC-FixedLR",
    "PALR-SAC-NoPerturb": "PALR-NoPerturb",
}

COLORS = {
    "PALR-SAC (ours)":    "#1f77b4",   # blue
    "PALR-SAC-NoScale":   "#ff7f0e",   # orange
    "SAC-FixedLR":        "#d62728",   # red
    "PALR-SAC-NoPerturb": "#7f7f7f",   # gray
}

LINESTYLES = {
    "PALR-SAC (ours)":    "-",
    "PALR-SAC-NoScale":   "-",
    "SAC-FixedLR":        "--",
    "PALR-SAC-NoPerturb": "-",
}

LINEWIDTHS = {
    "PALR-SAC (ours)": 2.5,
}

TASK_SHORT = [
    "Reach", "Push", "Pick Place", "Door Open", "Drawer Open",
    "Drawer Close", "Button Press", "Peg Insert", "Window Open", "Window Close",
]


def smooth(x, w=15):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_hero():
    with open(os.path.join(RESULTS_DIR, "cw_raw_results.json")) as f:
        raw = json.load(f)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    first_key   = list(raw.keys())[0]
    switch_eps  = raw[first_key][0]["task_switch_episodes"]
    n_eps       = len(raw[first_key][0]["episode_rewards"])
    ep_per_task = n_eps // len(CW10_TASKS)

    for sw in switch_eps:
        for ax in axes:
            ax.axvline(sw, color="gray", linestyle="--", lw=0.8, alpha=0.6)

    # ── Panel 1: Smoothed episode reward ──────────────────────────────────────
    for name in AGENTS:
        if name not in raw:
            continue
        rewards = np.array([r["episode_rewards"] for r in raw[name]])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r)
        sm_std  = smooth(std_r)
        x     = np.arange(len(sm_mean)) + 7
        color = COLORS[name]
        lw    = LINEWIDTHS.get(name, 1.8)
        ls    = LINESTYLES[name]
        axes[0].plot(x, sm_mean, label=DISPLAY[name], color=color,
                     linestyle=ls, linewidth=lw)
        axes[0].fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                             alpha=0.12, color=color)

    axes[0].set_ylabel("Episode Reward", fontsize=12)
    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # ── Panel 2: Layer-4 dead-neuron fraction ─────────────────────────────────
    sac_ctrs_for_annot, sac_dead_for_annot = [], []

    for name in AGENTS:
        if name not in raw:
            continue
        all_eps, all_dead = [], []
        for run in raw[name]:
            for entry in run["plasticity_log"]:
                all_eps.append(entry["episode"])
                all_dead.append(entry.get("layer_3_dead", 0.0))

        if not all_eps:
            continue
        max_ep = max(all_eps)
        bins   = np.arange(0, max_ep + 6, 5)
        dead_b, ctrs = [], []
        for b in range(len(bins) - 1):
            mask = [bins[b] <= e < bins[b + 1] for e in all_eps]
            if any(mask):
                dead_b.append(np.mean([all_dead[j] for j, m in enumerate(mask) if m]))
                ctrs.append((bins[b] + bins[b + 1]) / 2)

        color = COLORS[name]
        lw    = LINEWIDTHS.get(name, 1.8)
        ls    = LINESTYLES[name]
        axes[1].plot(ctrs, dead_b, color=color, linestyle=ls, linewidth=lw)
        axes[1].fill_between(ctrs, 0, dead_b, alpha=0.10, color=color)

        if name == "SAC-FixedLR":
            sac_ctrs_for_annot = ctrs
            sac_dead_for_annot = dead_b

    # Annotation pointing to SAC-FixedLR plateau
    if sac_ctrs_for_annot:
        ann_idx = int(len(sac_ctrs_for_annot) * 0.55)
        ax_y    = sac_dead_for_annot[ann_idx]
        ax_x    = sac_ctrs_for_annot[ann_idx]
        axes[1].annotate(
            "LR-only dead\nneurons accumulate",
            xy=(ax_x, ax_y),
            xytext=(ax_x + 200, ax_y - 0.12),
            fontsize=8, color=COLORS["SAC-FixedLR"],
            arrowprops=dict(arrowstyle="->", color=COLORS["SAC-FixedLR"], lw=1.0),
        )

    axes[1].set_ylabel("L4 Dead Fraction", fontsize=12)
    axes[1].set_xlabel("Episode", fontsize=12)
    axes[1].set_ylim(bottom=0)
    axes[1].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    axes[1].grid(True, alpha=0.3)

    # Task names at the bottom of panel 2, horizontal
    trans = blended_transform_factory(axes[1].transData, axes[1].transAxes)
    for i, tname in enumerate(TASK_SHORT):
        mid = (i * ep_per_task + (i + 1) * ep_per_task) / 2
        axes[1].text(mid, -0.22, tname, ha="center", va="top",
                     fontsize=7, color="gray", style="italic",
                     transform=trans, clip_on=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)
    path = os.path.join(PLOTS_DIR, "fig_paper1_hero.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_hero()
