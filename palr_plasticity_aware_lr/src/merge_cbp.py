"""
Merge CBP-SAC results into the existing CW10 results.
===================================================
The CBP baseline (Continual Backprop, Dohare et al. 2024) was run separately,
one checkpoint per seed:  cw_checkpoint_cbp_seed{N}.json  (keyed by "CBP-SAC").

This script loads the existing cw_raw_results.json (all other agents), replaces
/ inserts the "CBP-SAC" entry with the concatenated per-seed CBP runs, rewrites
cw_raw_results.json, and regenerates cw_summary_stats.json — WITHOUT touching any
other agent's runs.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/merge_cbp.py
"""
import os, sys, glob, json

sys.path.insert(0, os.path.dirname(__file__))
from run_cw_experiments import save_results, print_table   # reuse summary logic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EP_PER_TASK = 200   # matches the CBP run and the paper's CW10 budget


def main():
    raw_path = os.path.join(RESULTS_DIR, "cw_raw_results.json")
    with open(raw_path) as f:
        raw = json.load(f)
    print(f"Loaded cw_raw_results.json: "
          f"{[(k, len(v)) for k, v in raw.items()]}")

    cbp_runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              "cw_checkpoint_cbp_seed*.json"))):
        with open(path) as f:
            data = json.load(f)
        runs = data.get("CBP-SAC", [])
        cbp_runs.extend(runs)
        print(f"  {os.path.basename(path)}: CBP-SAC={len(runs)} run(s)")

    if not cbp_runs:
        print("No CBP checkpoint files found — nothing merged.")
        return

    raw["CBP-SAC"] = cbp_runs
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nMerged CBP-SAC ({len(cbp_runs)} runs) → {raw_path}")
    print(f"Agents now: {[(k, len(v)) for k, v in raw.items()]}")

    # Regenerate summary + print the CW10 table (includes CBP-SAC).
    summary = save_results(raw, EP_PER_TASK)
    print_table(summary)


if __name__ == "__main__":
    main()
