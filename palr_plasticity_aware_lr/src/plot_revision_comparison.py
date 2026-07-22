"""
RLC 2026 Revision: Full Agent Comparison Figure
=================================================
Generates fig_revision_comparison.png — shows ALL agents including the
four new revision baselines (CBP-SAC, ReDo-SAC, PALR-SAC-BetaOn,
PALR-SAC-NoOracle, PALR-SAC-AutoOracle).

Does NOT overwrite any existing figures.

Usage
-----
    bash run.sh palr_plasticity_aware_lr/src/plot_revision_comparison.py
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

# All agents — skip ones absent from the data (no error).
ALL_AGENTS = [
    "SAC-FixedLR",
    "SAC-ShrinkAndPerturb",
    "SAC-PeriodicReset",
    "SAC-L2Reg",
    "CBP-SAC",
    "ReDo-SAC",
    "PALR-SAC (ours)",
    "PALR-SAC-BetaOn",
    "PALR-SAC-NoOracle",
    "PALR-SAC-AutoOracle",
]

COLORS = {
    "SAC-FixedLR":            "#d62728",
    "SAC-ShrinkAndPerturb":   "#ff7f0e",
    "SAC-PeriodicReset":      "#9467bd",
    "SAC-L2Reg":              "#8c564b",
    "CBP-SAC":                "#e377c2",
    "ReDo-SAC":               "#17becf",
    "PALR-SAC (ours)":        "#2ca02c",
    "PALR-SAC-BetaOn":        "#bcbd22",
    "PALR-SAC-NoOracle":      "#aec7e8",
    "PALR-SAC-AutoOracle":    "#1f77b4",
}
LINESTYLES = {
    "SAC-FixedLR":            "--",
    "SAC-ShrinkAndPerturb":   ":",
    "SAC-PeriodicReset":      "-.",
    "SAC-L2Reg":              (0, (3, 1, 1, 1)),
    "CBP-SAC":                "-.",
    "ReDo-SAC":               ":",
    "PALR-SAC (ours)":        "-",
    "PALR-SAC-BetaOn":        "--",
    "PALR-SAC-NoOracle":      "--",
    "PALR-SAC-AutoOracle":    "-",
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


def plot_revision_comparison(raw, summary, window=20):
    present = [a for a in ALL_AGENTS if a in raw]
    print(f"  Agents present: {present}")
    absent  = [a for a in ALL_AGENTS if a not in raw]
    if absent:
        print(f"  Agents absent (skipped): {absent}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # ── Left: Learning curves ─────────────────────────────────────────────────
    ax = axes[0]
    ref_key = list(raw.keys())[0]
    switch_eps  = raw[ref_key][0]["task_switch_episodes"]
    n_eps       = len(raw[ref_key][0]["episode_rewards"])
    ep_per_task = n_eps // len(CW10_TASKS)

    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", lw=0.7, alpha=0.4,
                   label="Task switch" if i == 0 else None)

    for i, tname in enumerate(TASK_SHORT):
        s = i * ep_per_task
        e = (i + 1) * ep_per_task
        ax.text((s + e) / 2, -80, tname, ha="center", va="top",
                fontsize=6, color="gray", style="italic")

    for name in present:
        runs    = raw[name]
        rewards = np.array([r["episode_rewards"] for r in runs])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        x       = np.arange(len(sm_mean)) + window // 2
        color   = COLORS[name]
        lw      = 2.8 if "ours" in name else (2.2 if "AutoOracle" in name else 1.6)
        ax.plot(x, sm_mean, label=f"{name} (n={len(runs)})",
                color=color, linestyle=LINESTYLES[name], linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.10, color=color)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=11)
    ax.set_title("CW10: Learning Curves — All Agents", fontsize=12)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: Overall mean reward bar chart ──────────────────────────────────
    ax = axes[1]
    overall_means  = []
    overall_stderrs = []
    labels         = []
    colors         = []

    for name in present:
        if name not in summary:
            continue
        overall_means.append(summary[name].get("overall_mean_reward", 0.0))
        # Use SEM across seeds if available, else 0
        runs_arr = np.array([r["episode_rewards"] for r in raw[name]])
        seed_means = runs_arr.mean(axis=1)
        stderr = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0.0
        overall_stderrs.append(stderr)
        labels.append(name)
        colors.append(COLORS[name])

    x = np.arange(len(labels))
    bars = ax.bar(x, overall_means, color=colors, alpha=0.85,
                  yerr=overall_stderrs, capsize=4,
                  error_kw={"linewidth": 1.2})

    for bar, val in zip(bars, overall_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Overall Mean Episode Reward", fontsize=11)
    ax.set_title("CW10: Overall Score — All Agents", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_revision_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading CW10 results...")
    raw, summary = load()
    print(f"Agents in data: {list(raw.keys())}")
    print("Generating revision comparison figure...")
    plot_revision_comparison(raw, summary)
    print("Done.")
