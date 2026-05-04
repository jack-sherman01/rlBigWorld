"""
Plot PALR hyperparameter tuning results as a heatmap.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/plot_palr_tune.py
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
TUNE_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "tune")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_tune_results():
    files = sorted(glob.glob(os.path.join(TUNE_DIR, "palr_tune_*.json")))
    results = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        rewards = np.array(d["episode_rewards"])
        n_eps = len(rewards)
        results.append({
            "beta":          d["beta"],
            "sigma":         d["perturb_sigma"],
            "overall_mean":  float(rewards.mean()),
            "last_half_mean":float(rewards[n_eps // 2:].mean()),
        })
        print(f"  beta={d['beta']:<4}  sigma={d['perturb_sigma']:<4}  "
              f"overall={rewards.mean():.1f}  last_half={rewards[n_eps//2:].mean():.1f}")
    return results


def plot_heatmap(results):
    betas  = sorted(set(r["beta"]  for r in results))
    sigmas = sorted(set(r["sigma"] for r in results))

    overall   = np.full((len(betas), len(sigmas)), np.nan)
    last_half = np.full((len(betas), len(sigmas)), np.nan)

    for r in results:
        i = betas.index(r["beta"])
        j = sigmas.index(r["sigma"])
        overall[i, j]   = r["overall_mean"]
        last_half[i, j] = r["last_half_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title in zip(
        axes,
        [overall, last_half],
        ["Overall Mean Reward", "Last-Half Mean Reward\n(later tasks)"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"σ={s}" for s in sigmas], fontsize=10)
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"β={b}" for b in betas], fontsize=10)
        ax.set_xlabel("perturb_sigma", fontsize=11)
        ax.set_ylabel("beta", fontsize=11)
        ax.set_title(f"PALR Tuning: {title}", fontsize=11)

        # Annotate cells
        for i in range(len(betas)):
            for j in range(len(sigmas)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center",
                            fontsize=9, color="black", fontweight="bold")

    # Mark current config (beta=3.0 not in grid but note it)
    fig.suptitle("PALR Hyperparameter Grid Search (50 ep/task, 1 seed)\n"
                 "Current config: β=3.0, σ=0.3", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_palr_tune_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Loading tuning results...")
    results = load_tune_results()
    if not results:
        print("No results found yet.")
    else:
        print(f"\n{len(results)}/12 combos complete.")
        plot_heatmap(results)
