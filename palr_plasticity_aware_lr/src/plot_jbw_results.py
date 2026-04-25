"""
Plot JBW Experiment Results
===========================
Generates figures from jbw_raw_results.json / jbw_summary_stats.json.

Figures produced
----------------
  fig_jbw1_learning_curves.png   — smoothed reward over episodes, all agents
  fig_jbw2_phase_bars.png        — per-phase mean reward bar chart
  fig_jbw3_plasticity.png        — dead neuron fraction + effective rank
  fig_jbw4_ablation.png          — final-20ep reward ablation bar chart
  fig_jbw5_lr_control_loop.png   — LR scale + rank + dead (PALR control loop)

Usage
-----
    python plot_jbw_results.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "DQN-FixedLR":       "#d62728",
    "ShrinkAndPerturb":  "#ff7f0e",
    "PeriodicReset":     "#9467bd",
    "L2-Regularisation": "#8c564b",
    "PALR (ours)":       "#2ca02c",
    "PALR-NoScale":      "#17becf",
    "PALR-NoPerturb":    "#1f77b4",
}
LINESTYLES = {
    "DQN-FixedLR":       "-",
    "ShrinkAndPerturb":  "--",
    "PeriodicReset":     "-.",
    "L2-Regularisation": ":",
    "PALR (ours)":       "-",
    "PALR-NoScale":      "--",
    "PALR-NoPerturb":    ":",
}


def smooth(x, w=15):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load():
    with open(os.path.join(RESULTS_DIR, "jbw_raw_results.json")) as f:
        raw = json.load(f)
    with open(os.path.join(RESULTS_DIR, "jbw_summary_stats.json")) as f:
        summary = json.load(f)
    return raw, summary


# ---------------------------------------------------------------------------
# Fig 1: Learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(raw, summary, window=15):
    fig, ax = plt.subplots(figsize=(12, 5))

    switch_eps = raw[list(raw.keys())[0]][0]["task_switch_episodes"]
    for i, sw in enumerate(switch_eps):
        ax.axvline(sw, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Phase switch" if i == 0 else None)

    n_eps = len(raw[list(raw.keys())[0]][0]["episode_rewards"])
    phase_starts = [0] + switch_eps
    phase_ends   = switch_eps + [n_eps]
    for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
        sign = "Normal" if i % 2 == 0 else "Inverted"
        ax.text((s + e) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] > -999 else 0,
                f"Ph.{i+1}\n({sign})",
                ha="center", va="bottom", fontsize=7, color="gray", style="italic")

    for name, runs in raw.items():
        rewards = np.array([r["episode_rewards"] for r in runs])
        mean_r  = rewards.mean(axis=0)
        std_r   = rewards.std(axis=0)
        sm_mean = smooth(mean_r, window)
        sm_std  = smooth(std_r,  window)
        x = np.arange(len(sm_mean)) + window // 2
        color = COLORS.get(name, "black")
        lw    = 2.5 if name == "PALR (ours)" else 1.5
        ax.plot(x, sm_mean, label=name, color=color,
                linestyle=LINESTYLES.get(name, "-"), linewidth=lw)
        ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std,
                        alpha=0.10, color=color)

    # Re-draw phase labels after ylim is set
    ymin = ax.get_ylim()[0]
    for i, (s, e) in enumerate(zip(phase_starts, phase_ends)):
        sign = "Normal" if i % 2 == 0 else "Inverted"
        ax.text((s + e) / 2, ymin,
                f"Ph.{i+1} ({sign})",
                ha="center", va="bottom", fontsize=7.5, color="gray", style="italic")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.set_title("JellyBeanWorld (Continual): Learning Curves\n"
                 "Reward inverts every phase — no oracle given to agent",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw1_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 2: Per-phase bar chart
# ---------------------------------------------------------------------------
def plot_phase_bars(summary):
    agents  = list(summary.keys())
    sample  = list(summary.values())[0]["phase_rewards"]
    phases  = sorted(sample.keys())
    n_agents = len(agents)
    x = np.arange(len(phases))
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(agents):
        means = [summary[name]["phase_rewards"][p]["mean"] for p in phases]
        stds  = [summary[name]["phase_rewards"][p]["std"]  for p in phases]
        offset = (i - n_agents / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=name,
               color=COLORS.get(name, "gray"), alpha=0.82, capsize=3)

    phase_labels = [f"Ph.{i+1}\n({'Normal' if i%2==0 else 'Inverted'})"
                    for i in range(len(phases))]
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, fontsize=9)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("JBW: Per-Phase Performance (reward inverts each phase)", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw2_phase_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 3: Plasticity dynamics — dead fraction + effective rank
# ---------------------------------------------------------------------------
def plot_plasticity(raw):
    agents_to_plot = ["DQN-FixedLR", "PALR (ours)"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for name in agents_to_plot:
        if name not in raw:
            continue
        runs = raw[name]
        all_steps, all_dead, all_erank = [], [], []
        for run in runs:
            for entry in run["plasticity_log"]:
                all_steps.append(entry["episode"])
                all_dead.append(entry.get("mean_dead", 0.0))
                all_erank.append(entry.get("mean_erank", 1.0))

        if not all_steps:
            continue
        max_ep = max(all_steps)
        bins   = np.arange(0, max_ep + 5, 5)
        dead_b, erank_b, centers = [], [], []
        for b in range(len(bins) - 1):
            mask = [(bins[b] <= s < bins[b+1]) for s in all_steps]
            if any(mask):
                dead_b.append(np.mean([all_dead[j]  for j, m in enumerate(mask) if m]))
                erank_b.append(np.mean([all_erank[j] for j, m in enumerate(mask) if m]))
                centers.append((bins[b] + bins[b+1]) / 2)

        color = COLORS.get(name, "black")
        lw    = 2.5 if name == "PALR (ours)" else 1.5
        axes[0].plot(centers, dead_b,  label=name, color=color, linewidth=lw)
        axes[1].plot(centers, erank_b, label=name, color=color, linewidth=lw)

    switch_eps = raw[list(raw.keys())[0]][0]["task_switch_episodes"]
    for sw in switch_eps:
        axes[0].axvline(sw, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        axes[1].axvline(sw, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    axes[0].set_ylabel("Dead Neuron Fraction ↓", fontsize=11)
    axes[0].set_title("JBW Plasticity Dynamics: PALR vs DQN", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Effective Rank ↑", fontsize=11)
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw3_plasticity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 4: Ablation bar chart
# ---------------------------------------------------------------------------
def plot_ablation(summary):
    ablation = ["DQN-FixedLR", "PALR-NoPerturb", "PALR-NoScale", "PALR (ours)"]
    labels   = ["DQN\n(baseline)", "PALR\n(LR-only)", "PALR\n(Perturb-only)", "PALR\n(Full, ours)"]
    means, stds, colors = [], [], []
    for name in ablation:
        if name in summary:
            means.append(summary[name]["final_20ep_mean"])
            stds.append(summary[name]["final_20ep_std"])
        else:
            means.append(0); stds.append(0)
        colors.append(COLORS.get(name, "gray"))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(labels)), means, yerr=stds,
                  color=colors, alpha=0.85, capsize=5, width=0.55)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Final 20-Episode Mean Reward", fontsize=12)
    ax.set_title("JBW Ablation: PALR Components", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw4_ablation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 5: LR scale control loop (PALR only)
# ---------------------------------------------------------------------------
def plot_lr_control_loop(raw):
    palr_runs = raw.get("PALR (ours)", [])
    if not palr_runs or not palr_runs[0].get("palr_plasticity_history"):
        print("No PALR plasticity history — skipping fig_jbw5.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    layer_colors = {"l0": "#1f77b4", "l1": "#d62728"}

    for run in palr_runs:
        hist = run.get("palr_plasticity_history", [])
        steps    = [h["step"]        for h in hist]
        scale_l0 = [h["lr_scale_l0"] for h in hist]
        scale_l1 = [h["lr_scale_l1"] for h in hist]
        erank_l0 = [h["erank_l0"]    for h in hist]
        erank_l1 = [h["erank_l1"]    for h in hist]
        dead_l0  = [h["dead_l0"]     for h in hist]
        dead_l1  = [h["dead_l1"]     for h in hist]
        axes[0].plot(steps, scale_l0, color=layer_colors["l0"], alpha=0.7, lw=1.2, label="Layer 1")
        axes[0].plot(steps, scale_l1, color=layer_colors["l1"], alpha=0.7, lw=1.2, label="Layer 2")
        axes[1].plot(steps, erank_l0, color=layer_colors["l0"], alpha=0.7, lw=1.2)
        axes[1].plot(steps, erank_l1, color=layer_colors["l1"], alpha=0.7, lw=1.2)
        axes[2].plot(steps, dead_l0,  color=layer_colors["l0"], alpha=0.7, lw=1.2)
        axes[2].plot(steps, dead_l1,  color=layer_colors["l1"], alpha=0.7, lw=1.2)

    # Phase boundary lines (approximate)
    switch_eps = palr_runs[0].get("task_switch_episodes", [])
    hist = palr_runs[0].get("palr_plasticity_history", [])
    if hist and switch_eps:
        total_steps  = hist[-1]["step"]
        n_eps = len(palr_runs[0].get("episode_rewards", [])) or 1
        steps_per_ep = total_steps / n_eps
        for sw in switch_eps:
            for ax in axes:
                ax.axvline(sw * steps_per_ep, color="gray",
                           linestyle="--", linewidth=0.9, alpha=0.7)

    axes[0].axhline(1.0, color="black", linestyle=":", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("LR Scale $s^{(l)}$", fontsize=11)
    axes[0].set_title("JBW Control Loop: Rank Deficit → LR Scale (PALR)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Effective Rank $\\rho^{(l)}$", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("Dead Fraction $\\delta^{(l)}$", fontsize=11)
    axes[2].set_xlabel("Training Step", fontsize=11)
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_jbw5_lr_control_loop.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading JBW results...")
    raw, summary = load()
    print("Generating figures...")
    plot_learning_curves(raw, summary)
    plot_phase_bars(summary)
    plot_plasticity(raw)
    plot_ablation(summary)
    plot_lr_control_loop(raw)
    print(f"\nAll JBW figures saved to: {PLOTS_DIR}")
