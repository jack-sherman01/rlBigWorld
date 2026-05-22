"""
JBW Task Visualization
======================
Creates two publication-quality figures illustrating the JellyBeanWorld
continual RL benchmark:

  fig_jbw_task_overview.png  — phase timeline + item legend + reward-sign flip
  fig_jbw_agent_vision.png   — 4 representative agent vision frames (one per phase)

These use the saved MP4 videos in plots/jbw_videos/ — no live JBW environment
needed.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_jbw_tasks.py
    # or directly:
    python palr_plasticity_aware_lr/src/plot_jbw_tasks.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.ticker import MultipleLocator

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "..", "plots")
VIDEOS_DIR  = os.path.join(PLOTS_DIR,  "jbw_videos")
OUT_OVERVIEW = os.path.join(PLOTS_DIR, "fig_jbw_task_overview.png")
OUT_VISION   = os.path.join(PLOTS_DIR, "fig_jbw_agent_vision.png")

# ── Palette (matched to actual JBW render colours) ──────────────────────────
COL_BG        = "#0d0d0d"    # near-black floor
COL_JELLYBEAN = "#22cc22"    # green
COL_ONION     = "#1a5aff"    # blue (JBW render)
COL_WALL      = "#7e7e7e"    # grey
COL_BANANA    = "#e8c200"    # yellow (rarely seen)

# Phase colours for the timeline ribbon
PHASE_COLS = ["#d4eaff", "#ffd4d4", "#d4eaff", "#ffd4d4"]
PHASE_LABELS = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
SIGN_LABELS  = ["+1 jellybean / −1 onion",
                "−1 jellybean / +1 onion",
                "+1 jellybean / −1 onion",
                "−1 jellybean / +1 onion"]


# ════════════════════════════════════════════════════════════════════════════
# Helper: load a single representative frame from a video
# ════════════════════════════════════════════════════════════════════════════
def load_frame(video_path: str, frame_idx: int = 400) -> np.ndarray:
    """Load one frame from an MP4 as (220, 220, 3) uint8."""
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        frame  = reader.get_data(frame_idx)
        reader.close()
        return frame
    except Exception as e:
        print(f"  [warn] Could not read {video_path}: {e}")
        return np.zeros((220, 220, 3), dtype=np.uint8)


def richest_frame(video_path: str) -> np.ndarray:
    """Return the frame with the most visible (non-black) content."""
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        best_frame_data = None
        best_count = -1
        for i in range(0, 1000, 20):
            try:
                frame = reader.get_data(i)
                bright = int(np.sum(np.max(frame, axis=2) > 80))
                if bright > best_count:
                    best_count = bright
                    best_frame_data = frame.copy()
            except Exception:
                break
        reader.close()
        return best_frame_data if best_frame_data is not None else np.zeros((220, 220, 3), dtype=np.uint8)
    except Exception as e:
        print(f"  [warn] Could not read {video_path}: {e}")
        return np.zeros((220, 220, 3), dtype=np.uint8)


def best_frame(seed: int, agent: str, episode: str, frame_idx: int = None) -> np.ndarray:
    """Try seed first; fall back to other seeds. Use richest frame.
    episode: e.g. 'ep0050' (already includes the 'ep' prefix)."""
    path = os.path.join(VIDEOS_DIR, f"seed{seed}", f"{episode}_{agent}_seed{seed}.mp4")
    if os.path.exists(path):
        return richest_frame(path)
    for s in range(5):
        p2 = os.path.join(VIDEOS_DIR, f"seed{s}", f"{episode}_{agent}_seed{s}.mp4")
        if os.path.exists(p2):
            return richest_frame(p2)
    return np.zeros((220, 220, 3), dtype=np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Task Overview
# ════════════════════════════════════════════════════════════════════════════
def plot_task_overview():
    fig = plt.figure(figsize=(10, 6.0))
    fig.patch.set_facecolor("white")

    # ── Layout: top = item legend; middle = phase timeline; bottom = note ──
    gs = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[2.0, 1.6, 0.5],
        hspace=0.6,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Row 0 — Item legend
    # ────────────────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[0])
    ax_leg.set_xlim(0, 10)
    ax_leg.set_ylim(-0.1, 1.3)
    ax_leg.axis("off")
    ax_leg.set_title("JellyBeanWorld: Items & Partial-Observation Setting",
                     fontsize=11, fontweight="bold", pad=8)

    items = [
        ("Jellybean",  COL_JELLYBEAN,  "reward ±1 (phase-dependent)", "circle"),
        ("Onion",      COL_ONION,      "reward ∓1 (phase-dependent)", "circle"),
        ("Wall",       COL_WALL,       "impassable  (reward = 0)",     "square"),
        ("Floor",      COL_BG,         "empty cell  (reward = 0)",     "square"),
    ]

    xs = [1.0, 3.5, 6.0, 8.5]
    for (name, color, desc, shape), x in zip(items, xs):
        # Coloured icon
        if shape == "circle":
            circ = plt.Circle((x, 0.78), 0.28, color=color,
                              ec="black", lw=0.8, zorder=3)
            ax_leg.add_patch(circ)
        else:
            sq = FancyBboxPatch((x - 0.28, 0.50), 0.56, 0.56,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="black", lw=0.8)
            ax_leg.add_patch(sq)
        ax_leg.text(x, 0.44, name, ha="center", va="top",
                    fontsize=10, fontweight="bold")
        ax_leg.text(x, 0.20, desc, ha="center", va="top",
                    fontsize=7.5, color="#555555", style="italic")

    # 11×11 vision box annotation on the right
    vis_x = 9.55
    ax_leg.text(vis_x, 1.10,
                "11×11 local\nvision patch",
                ha="center", va="top", fontsize=8, color="#222222",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0",
                          ec="#888888", lw=0.7))

    # ────────────────────────────────────────────────────────────────────────
    # Row 1 — Phase timeline ribbon
    # ────────────────────────────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1])
    ax_t.set_xlim(0, 400)
    ax_t.set_ylim(0, 1)
    ax_t.axis("off")
    ax_t.set_title("Non-stationary Reward Schedule  (no oracle — boundary invisible to agent)",
                   fontsize=9.5, pad=5, color="#333333")

    phase_starts = [0, 100, 200, 300]
    phase_width  = 100

    for i, (start, col) in enumerate(zip(phase_starts, PHASE_COLS)):
        # Phase ribbon
        rect = FancyBboxPatch((start, 0.30), phase_width, 0.55,
                              boxstyle="round,pad=1",
                              facecolor=col, edgecolor="#aaaaaa", lw=0.6,
                              transform=ax_t.transData, clip_on=False)
        ax_t.add_patch(rect)

        mid = start + phase_width / 2
        sign_positive = (i % 2 == 0)

        # Phase number
        ax_t.text(mid, 0.78, PHASE_LABELS[i],
                  ha="center", va="center", fontsize=9, fontweight="bold",
                  color="#222222")

        # Reward sign summary inside ribbon
        jb_col  = COL_JELLYBEAN if sign_positive else COL_ONION
        on_col  = COL_ONION     if sign_positive else COL_JELLYBEAN
        jb_sign = "+1" if sign_positive else "−1"
        on_sign = "−1" if sign_positive else "+1"

        # Small coloured dots + reward sign
        circ_jb = plt.Circle((mid - 24, 0.57), 0.045, color=COL_JELLYBEAN,
                              ec="black", lw=0.5, zorder=4,
                              transform=ax_t.transData)
        circ_on = plt.Circle((mid - 24, 0.42), 0.045, color=COL_ONION,
                              ec="black", lw=0.5, zorder=4,
                              transform=ax_t.transData)
        ax_t.add_patch(circ_jb)
        ax_t.add_patch(circ_on)

        ax_t.text(mid - 14, 0.57, f"Jellybean {jb_sign}",
                  ha="left", va="center", fontsize=7.5,
                  color=jb_col, fontweight="bold")
        ax_t.text(mid - 14, 0.42, f"Onion {on_sign}",
                  ha="left", va="center", fontsize=7.5,
                  color=on_col, fontweight="bold")

    # Episode axis
    ax_t.annotate("", xy=(405, 0.18), xytext=(-5, 0.18),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    for ep in range(0, 401, 100):
        ax_t.text(ep, 0.04, str(ep), ha="center", va="bottom",
                  fontsize=8, color="#444444")
        ax_t.plot([ep, ep], [0.18, 0.28], color="#888888", lw=0.8)
    ax_t.text(200, -0.12, "Episode", ha="center", va="bottom",
              fontsize=8.5, color="#444444")

    # "Phase switch" markers
    for ep in [100, 200, 300]:
        ax_t.annotate("",
                      xy=(ep, 0.30), xytext=(ep, 0.21),
                      arrowprops=dict(arrowstyle="-|>", color="#cc2222",
                                      lw=1.2))
    ax_t.text(200, 0.18, "← reward sign flips (unannounced) →",
              ha="center", va="top", fontsize=7.5, color="#cc2222",
              style="italic")

    # ────────────────────────────────────────────────────────────────────────
    # Row 2 — challenge note
    # ────────────────────────────────────────────────────────────────────────
    ax_note = fig.add_subplot(gs[2])
    ax_note.axis("off")
    ax_note.text(
        0.5, 0.6,
        "Challenge: optimal policy must reverse (seek ↔ avoid) at each phase boundary"
        " without any signal — purely from reward experience.",
        ha="center", va="center", fontsize=9, color="#333333",
        style="italic",
        transform=ax_note.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fffff0", ec="#cccc88", lw=0.8),
    )

    fig.savefig(OUT_OVERVIEW, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_OVERVIEW}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Agent Vision Across Phases
# ════════════════════════════════════════════════════════════════════════════
def plot_agent_vision():
    """
    4-column figure: one representative vision frame per phase.
    Uses PALR_ours agent, seed 0, mid-episode snapshots.
    Each panel is annotated with phase info and reward sign.
    """
    AGENT       = "PALR_ours"
    SEED        = 0
    FRAME_IDX   = 400   # mid-episode (out of 1000 frames)

    # Episodes: 50 = phase1, 150 = phase2, 250 = phase3, 350 = phase4
    EPISODES = [("ep0050", "Phase 1",  True),
                ("ep0150", "Phase 2",  False),
                ("ep0250", "Phase 3",  True),
                ("ep0350", "Phase 4",  False)]

    frames = []
    for ep_str, phase_name, sign_positive in EPISODES:
        frame = best_frame(SEED, AGENT, ep_str, FRAME_IDX)
        frames.append((frame, phase_name, sign_positive))

    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "PALR Agent's 11×11 Egocentric Vision  (upscaled for visibility)",
        fontsize=10.5, fontweight="bold", y=1.01
    )

    for ax, (frame, phase_name, sign_positive) in zip(axes, frames):
        ax.imshow(frame, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # Colour-coded border
        border_col = "#2266cc" if sign_positive else "#cc2222"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)

        # Phase title
        ax.set_title(phase_name, fontsize=9.5, fontweight="bold", pad=4)

        # Reward-sign box below image
        jb_sign = "+1" if sign_positive else "−1"
        on_sign = "−1" if sign_positive else "+1"
        label_col = "#2266cc" if sign_positive else "#cc2222"
        ax.text(
            0.5, -0.06,
            f"Jellybean {jb_sign}   Onion {on_sign}",
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=7.5, color=label_col, fontweight="bold",
        )

        # "Seek / Avoid" annotation
        seek_item  = "Jellybean" if sign_positive else "Onion"
        avoid_item = "Onion"     if sign_positive else "Jellybean"
        ax.text(
            0.5, -0.19,
            f"seek {seek_item} · avoid {avoid_item}",
            ha="center", va="top",
            transform=ax.transAxes,
            fontsize=7.5, color="#333333", style="italic",
        )

    # Legend row (bottom of figure)
    legend_patches = [
        mpatches.Patch(color=COL_JELLYBEAN, label="Jellybean"),
        mpatches.Patch(color=COL_ONION,     label="Onion"),
        mpatches.Patch(color=COL_WALL,      label="Wall"),
        mpatches.Patch(color=COL_BG,        label="Floor"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.18),
        title="Item colour key",
        title_fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_VISION, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_VISION}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Generating JBW task visualisation figures...")
    plot_task_overview()
    plot_agent_vision()
    print("Done.")
