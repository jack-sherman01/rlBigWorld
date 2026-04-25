"""
plot_results.py
===============
Visualise PALR Fetch Rearrangement results from TensorBoard event files
or the saved palr_plasticity_history.json / curriculum_switch_log.json.

Usage:
    conda activate palr_habitat
    python src/plot_results.py \\
        --palr_dir  results/palr_seed0 \\
        --base_dir  results/baseline_seed0 \\
        --outdir    plots/

Produces:
  Fig 1: Success rate over env steps (PALR vs baseline, task phases shaded)
  Fig 2: Per-block dead-filter fraction over training (4 lines)
  Fig 3: Per-block effective rank over training (4 lines)
  Fig 4: Per-block LR scale over training (PALR control loop)
  Fig 5: Task-phase success rate bar chart (early vs late phase per task)
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── TensorBoard reader ────────────────────────────────────────────────────────

def read_tb(tb_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a scalar tag from TensorBoard event files → (steps, values)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(x: np.ndarray, w: int = 20) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


# ── Colours ───────────────────────────────────────────────────────────────────

BLOCK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
BLOCK_LABELS = ["Block 0 (layer1, 64ch)", "Block 1 (layer2, 128ch)",
                "Block 2 (layer3, 256ch)", "Block 3 (layer4, 512ch)"]
PHASE_COLORS = ["#fff3b0", "#d4edda", "#d1ecf1", "#f8d7da"]
PHASE_LABELS = ["Apple Pick", "Bowl Pick", "Open Fridge", "Sink Place"]


def add_phase_spans(ax, switch_log: List[dict], total_steps: int, alpha=0.15):
    """Shade background by curriculum phase."""
    boundaries = [0] + [e["step"] for e in switch_log] + [total_steps]
    phases = [e["to"] for e in switch_log] if switch_log else []
    phase_order = [e.get("from", PHASE_LABELS[0]) for e in switch_log[:1]] + phases
    for i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        c = PHASE_COLORS[i % len(PHASE_COLORS)]
        ax.axvspan(lo, hi, alpha=alpha, color=c, label=PHASE_LABELS[i % 4] if i < 4 else "")


# ── Figure 1: Success rate ─────────────────────────────────────────────────────

def fig_success_rate(palr_tb: str, base_tb: str, switch_log: list, outdir: str):
    fig, ax = plt.subplots(figsize=(10, 4))

    steps_p, vals_p = read_tb(palr_tb, "train/success_rate")
    steps_b, vals_b = read_tb(base_tb,  "train/success_rate")

    if len(steps_p):
        ax.plot(steps_p, smooth(vals_p), color="#e41a1c", label="PALR (ours)")
    if len(steps_b):
        ax.plot(steps_b, smooth(vals_b), color="#999999", label="Baseline (fixed LR)",
                linestyle="--")

    total = max(steps_p.max() if len(steps_p) else 0,
                steps_b.max() if len(steps_b) else 0)
    add_phase_spans(ax, switch_log, total)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("Fetch Continual Curriculum — Success Rate")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig1_success_rate.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Dead-filter fraction per block ───────────────────────────────────

def fig_dead_fraction(history: list, switch_log: list, outdir: str):
    if not history:
        print("  [skip] No plasticity history found.")
        return

    steps  = np.array([e["update"] for e in history])
    fig, ax = plt.subplots(figsize=(10, 4))

    for k in range(4):
        key = f"block_{k}_dead"
        vals = np.array([e.get(key, np.nan) for e in history])
        ax.plot(steps, smooth(vals, 10), color=BLOCK_COLORS[k],
                label=BLOCK_LABELS[k], linewidth=1.8)

    ax.axhline(0.10, color="k", linestyle=":", linewidth=1, label="perturb threshold (10%)")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Dead-Filter Fraction")
    ax.set_title("ResNet-18 Dead-Filter Fraction per Block (PALR)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    path = os.path.join(outdir, "fig2_dead_filters.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Effective rank per block ────────────────────────────────────────

def fig_effective_rank(history: list, outdir: str):
    if not history:
        return

    steps = np.array([e["update"] for e in history])
    fig, ax = plt.subplots(figsize=(10, 4))

    baseline_erank = [48, 96, 192, 384]
    for k in range(4):
        key  = f"block_{k}_erank"
        vals = np.array([e.get(key, np.nan) for e in history])
        ax.plot(steps, smooth(vals, 10), color=BLOCK_COLORS[k],
                label=BLOCK_LABELS[k], linewidth=1.8)
        ax.axhline(baseline_erank[k], color=BLOCK_COLORS[k],
                   linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("PPO Update")
    ax.set_ylabel("Effective Rank")
    ax.set_title("ResNet-18 Effective Rank per Block (GAP + SVD)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, "fig3_effective_rank.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: LR scale per block (PALR control loop) ─────────────────────────

def fig_lr_scales(history: list, switch_log: list, outdir: str):
    if not history:
        return

    steps = np.array([e["update"] for e in history])
    fig, ax = plt.subplots(figsize=(10, 4))

    for k in range(4):
        key  = f"block_{k}_lr_scale"
        vals = np.array([e.get(key, 1.0) for e in history])
        ax.plot(steps, vals, color=BLOCK_COLORS[k],
                label=BLOCK_LABELS[k], linewidth=1.5)

    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label="scale=1 (no boost)")
    ax.set_xlabel("PPO Update")
    ax.set_ylabel("LR Scale")
    ax.set_title("PALR Per-Block Learning Rate Scale (ResNet-18 Backbone)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.8, 5.5)
    ax.grid(True, alpha=0.3)

    # Mark task switches
    for sw in switch_log:
        ax.axvline(sw.get("update", sw.get("step", 0)),
                   color="gray", linestyle=":", linewidth=0.8)

    path = os.path.join(outdir, "fig4_lr_scales.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Per-phase success (early vs late) ───────────────────────────────

def fig_phase_bar(palr_tb: str, base_tb: str, switch_log: list,
                  outdir: str, steps_per_phase: int = 50_000_000):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (tb_dir, label, color) in zip(
        axes,
        [(palr_tb, "PALR", "#e41a1c"), (base_tb, "Baseline", "#999999")]
    ):
        steps, vals = read_tb(tb_dir, "train/success_rate")
        if len(steps) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue

        phase_names = PHASE_LABELS
        early_means, late_means = [], []
        phase_boundaries = [i * steps_per_phase for i in range(5)]

        for i in range(4):
            lo, hi = phase_boundaries[i], phase_boundaries[i + 1]
            mid    = (lo + hi) // 2
            mask_e = (steps >= lo)  & (steps < mid)
            mask_l = (steps >= mid) & (steps < hi)
            early_means.append(vals[mask_e].mean() if mask_e.any() else 0.0)
            late_means.append(vals[mask_l].mean()  if mask_l.any() else 0.0)

        x = np.arange(4)
        ax.bar(x - 0.2, early_means, 0.35, label="Early (first half)", color=color, alpha=0.5)
        ax.bar(x + 0.2, late_means,  0.35, label="Late (second half)",  color=color, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Success Rate")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Task Success Rate: Early vs Late Phase")
    path = os.path.join(outdir, "fig5_phase_bar.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--palr_dir",  required=True, help="results/palr_seed0/")
    p.add_argument("--base_dir",  required=True, help="results/baseline_seed0/")
    p.add_argument("--outdir",    default="plots/")
    p.add_argument("--steps_per_phase", type=int, default=50_000_000)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    palr_tb = os.path.join(args.palr_dir, "tb")
    base_tb = os.path.join(args.base_dir, "tb")

    # Load plasticity history
    hist_path = os.path.join(args.palr_dir, "palr_plasticity_history.json")
    history   = json.load(open(hist_path)) if os.path.exists(hist_path) else []

    sw_path    = os.path.join(args.palr_dir, "curriculum_switch_log.json")
    switch_log = json.load(open(sw_path)) if os.path.exists(sw_path) else []

    print("Generating figures...")
    fig_success_rate(palr_tb, base_tb, switch_log, args.outdir)
    fig_dead_fraction(history, switch_log, args.outdir)
    fig_effective_rank(history, args.outdir)
    fig_lr_scales(history, switch_log, args.outdir)
    fig_phase_bar(palr_tb, base_tb, switch_log, args.outdir, args.steps_per_phase)

    print(f"\nAll figures → {args.outdir}")
    print("Open with:  evince plots/*.pdf  OR  open plots/  (macOS)")


if __name__ == "__main__":
    main()
