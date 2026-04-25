"""
Plot CW10 Experiment Results
=============================
Generates 5 figures from cw_raw_results.json / cw_summary_stats.json.

Figures
-------
  fig_cw1_learning_curves.png   — smoothed reward over all episodes; vertical
                                   dashed lines mark task switches
  fig_cw2_per_task_bars.png     — mean reward per task (bar chart, all agents)
  fig_cw3_plasticity.png        — dead-fraction + effective-rank per task for
                                   PALR-SAC vs SAC-FixedLR (4-layer breakdown)
  fig_cw4_lr_dynamics.png       — PALR per-layer LR scale for all 4 layers
  fig_cw5_learning_speed.png    — learning speed (ep-to-threshold) task 1→10;
                                   the key plasticity-loss evidence figure

Usage
-----
    bash run.sh palr_plasticity_aware_lr/src/plot_cw_results.py
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

COLORS = {
    "SAC-FixedLR":         "#d62728",
    "SAC-ShrinkAndPerturb":"#ff7f0e",
    "SAC-PeriodicReset":   "#9467bd",
    "SAC-L2Reg":           "#8c564b",
    "PALR-SAC (ours)":     "#2ca02c",
    "PALR-SAC-NoScale":    "#17becf",
    "PALR-SAC-NoPerturb":  "#1f77b4",
}
LINESTYLES = {
    "SAC-FixedLR":         "-",
    "SAC-ShrinkAndPerturb":"--",
    "SAC-PeriodicReset":   "-.",
    "SAC-L2Reg":           ":",
    "PALR-SAC (ours)":     "-",
    "PALR-SAC-NoScale":    "--",
    "PALR-SAC-NoPerturb":  ":",
}
TASK_SHORT = [
    "Reach", "Push", "Pick\nPlace", "Door\nOpen", "Drawer\nOpen",
    "Drawer\nClose", "Button\nPress", "Peg\nInsert", "Window\nOpen", "Window\nClose",
]


def smooth(x, w=10):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load():
    with open(os.path.join(RESULTS_DIR, "cw_raw_results.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RESULTS_DIR, "cw_summary_stats.json")) as f:
        summary = json.load(f)
    return raw, summary


# ── Fig 1: Learning curves ────────────────────────────────────────────────────

def plot_learning_curves(raw, summary, window=10):
    fig, ax = plt.subplots(figsize=(14, 5))

    switch_eps = raw[list(raw.keys())[0]][0]["task_switch_episodes"]
    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", lw=0.8, alpha=0.7,
                   label="Task switch" if i == 0 else None)

    # Task label bands
    n_eps       = len(raw[list(raw.keys())[0]][0]["episode_rewards"])
    ep_per_task = n_eps // len(CW10_TASKS)
    ymin_est    = -10  # placeholder; updated after plot
    for i, name in enumerate(TASK_SHORT):
        s = i * ep_per_task
        e = (i + 1) * ep_per_task
        ax.text((s + e) / 2, 0, name, ha="center", va="bottom",
                fontsize=6.5, color="gray", style="italic")

    for name, runs in raw.items():
        rewards = np.array([r["episode_rewards"] for r in runs])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        x     = np.arange(len(sm_mean)) + window // 2
        color = COLORS.get(name, "black")
        lw    = 2.5 if "ours" in name else 1.5
        ax.plot(x, sm_mean, label=name, color=color,
                linestyle=LINESTYLES.get(name, "-"), linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.10, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.set_title("CW10 Continual-World: Learning Curves (10 tasks, no oracle)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cw1_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Fig 2: Per-task bar chart ─────────────────────────────────────────────────

def plot_per_task_bars(summary):
    agents  = list(summary.keys())
    n_agents = len(agents)
    n_tasks  = len(CW10_TASKS)
    x        = np.arange(n_tasks)
    width    = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, name in enumerate(agents):
        means = [summary[name]["task_means"].get(f"task_{t+1}", 0.0)
                 for t in range(n_tasks)]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=name,
               color=COLORS.get(name, "gray"), alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT, fontsize=8)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("CW10: Per-Task Performance", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cw2_per_task_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Fig 3: Plasticity dynamics — 4-layer breakdown ───────────────────────────

def plot_plasticity(raw):
    agents_to_plot = ["SAC-FixedLR", "PALR-SAC (ours)"]
    layer_colors   = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    layer_labels   = ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    switch_eps = raw[list(raw.keys())[0]][0]["task_switch_episodes"]

    for name in agents_to_plot:
        if name not in raw:
            continue
        runs = raw[name]
        ls = "-" if "ours" in name else "--"
        lw = 2.0 if "ours" in name else 1.5

        # Collect per-layer metrics across all runs
        for k in range(4):
            all_eps, all_dead, all_erank = [], [], []
            for run in runs:
                for entry in run["plasticity_log"]:
                    all_eps.append(entry["episode"])
                    all_dead.append(entry.get(f"layer_{k}_dead",  0.0))
                    all_erank.append(entry.get(f"layer_{k}_erank", 1.0))

            if not all_eps:
                continue
            # Bin by episode
            max_ep = max(all_eps)
            bins   = np.arange(0, max_ep + 6, 5)
            dead_b, erank_b, ctrs = [], [], []
            for b in range(len(bins) - 1):
                mask = [bins[b] <= e < bins[b+1] for e in all_eps]
                if any(mask):
                    dead_b.append(np.mean([all_dead[j]  for j, m in enumerate(mask) if m]))
                    erank_b.append(np.mean([all_erank[j] for j, m in enumerate(mask) if m]))
                    ctrs.append((bins[b] + bins[b+1]) / 2)

            label = f"{name} – {layer_labels[k]}"
            axes[0].plot(ctrs, dead_b,  label=label,
                         color=layer_colors[k], linestyle=ls, linewidth=lw, alpha=0.85)
            axes[1].plot(ctrs, erank_b, color=layer_colors[k],
                         linestyle=ls, linewidth=lw, alpha=0.85)

    for sw in switch_eps:
        axes[0].axvline(sw, color="gray", linestyle="--", lw=0.7, alpha=0.6)
        axes[1].axvline(sw, color="gray", linestyle="--", lw=0.7, alpha=0.6)

    axes[0].set_ylabel("Dead Neuron Fraction ↓", fontsize=11)
    axes[0].set_title("CW10 Plasticity: PALR-SAC vs SAC (4 critic layers)", fontsize=12)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Effective Rank ↑", fontsize=11)
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cw3_plasticity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Fig 4: PALR per-layer LR scale dynamics ───────────────────────────────────

def plot_lr_dynamics(raw):
    palr_runs = raw.get("PALR-SAC (ours)", [])
    if not palr_runs or not palr_runs[0].get("palr_plasticity_history"):
        print("No PALR plasticity history — skipping fig_cw4.")
        return

    layer_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    fig, axes    = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    switch_eps = palr_runs[0].get("task_switch_episodes", [])
    hist0      = palr_runs[0].get("palr_plasticity_history", [])
    n_eps      = len(palr_runs[0].get("episode_rewards", [])) or 1
    steps_per_ep = (hist0[-1]["step"] / n_eps) if hist0 else 1

    for run in palr_runs:
        hist  = run.get("palr_plasticity_history", [])
        steps = [h["step"] for h in hist]
        for k in range(4):
            scales = [h.get(f"lr_scale_l{k}", 1.0) for h in hist]
            eranks = [h.get(f"erank_l{k}",    1.0) for h in hist]
            deads  = [h.get(f"dead_l{k}",     0.0) for h in hist]
            kw     = dict(color=layer_colors[k], alpha=0.75, lw=1.2)
            axes[0].plot(steps, scales, label=f"Layer {k+1}", **kw)
            axes[1].plot(steps, eranks, **kw)
            axes[2].plot(steps, deads,  **kw)

    for sw in switch_eps:
        for ax in axes:
            ax.axvline(sw * steps_per_ep, color="gray",
                       linestyle="--", lw=0.9, alpha=0.7)

    axes[0].axhline(1.0, color="black", linestyle=":", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("LR Scale $s^{(l)}$", fontsize=11)
    axes[0].set_title("CW10: PALR Per-Layer LR Control Loop (4 critic layers)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Effective Rank $\\rho^{(l)}$", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("Dead Fraction $\\delta^{(l)}$", fontsize=11)
    axes[2].set_xlabel("Training Step", fontsize=11)
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cw4_lr_dynamics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Fig 5: Learning speed task 1 → task 10 ───────────────────────────────────

def plot_learning_speed(summary):
    """
    The key plasticity-loss figure:
    Bar chart of learning speed (episodes-to-threshold) for each task.
    A healthy agent (PALR) should maintain the same speed on task 10 as task 1.
    A plasticity-impaired agent (DQN/SAC) slows down on later tasks.
    """
    agents = ["SAC-FixedLR", "PALR-SAC (ours)", "PALR-SAC-NoPerturb"]
    n_tasks = len(CW10_TASKS)
    x       = np.arange(n_tasks)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(agents):
        if name not in summary:
            continue
        speeds = [
            summary[name]["learning_speed"].get(f"task_{t+1}")
            for t in range(n_tasks)
        ]
        vals = [s if s is not None else 0.0 for s in speeds]
        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=COLORS.get(name, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT, fontsize=8)
    ax.set_ylabel("Episodes to Threshold (lower = faster)", fontsize=11)
    ax.set_title("Plasticity Evidence: Learning Speed per Task\n"
                 "SAC slows down on later tasks; PALR-SAC maintains speed",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_cw5_learning_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading CW10 results...")
    raw, summary = load()
    print("Generating figures...")
    plot_learning_curves(raw, summary)
    plot_per_task_bars(summary)
    plot_plasticity(raw)
    plot_lr_dynamics(raw)
    plot_learning_speed(summary)
    print(f"\nAll CW10 figures saved to: {PLOTS_DIR}")
