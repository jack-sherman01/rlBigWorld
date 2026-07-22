"""
Merge ReDo-SAC, PALR-SAC-BetaOn, PALR-SAC-NoOracle, and PALR-SAC-AutoOracle results into CW10.
==============================================================================================
These agents were run for the RLC 2026 workshop revision. Each has 3 seeds
saved as cw_checkpoint_{redo,betaon,nooracle,autooracle}_seed{N}.json.

Merges into cw_raw_results.json and regenerates cw_summary_stats.json
WITHOUT touching any other agent's runs.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/merge_revision_experiments.py
"""
import os, sys, glob, json

sys.path.insert(0, os.path.dirname(__file__))
from run_cw_experiments import save_results, print_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
EP_PER_TASK = 200

AGENTS = [
    ("ReDo-SAC",              "redo"),
    ("PALR-SAC-BetaOn",       "betaon"),
    ("PALR-SAC-NoOracle",     "nooracle"),
    ("PALR-SAC-AutoOracle",   "autooracle"),
]


def main():
    raw_path = os.path.join(RESULTS_DIR, "cw_raw_results.json")
    with open(raw_path) as f:
        raw = json.load(f)
    print(f"Loaded cw_raw_results.json: {[(k, len(v)) for k, v in raw.items()]}")

    for agent_name, suffix in AGENTS:
        pattern = os.path.join(RESULTS_DIR, f"cw_checkpoint_{suffix}_seed*.json")
        paths   = sorted(glob.glob(pattern))
        if not paths:
            print(f"\n  WARNING: No checkpoint files found for {agent_name} "
                  f"(pattern: cw_checkpoint_{suffix}_seed*.json) — skipping.")
            continue

        runs = []
        for path in paths:
            with open(path) as f:
                data = json.load(f)
            agent_runs = data.get(agent_name, [])
            runs.extend(agent_runs)
            print(f"  {os.path.basename(path)}: {agent_name}={len(agent_runs)} run(s)")

        if runs:
            raw[agent_name] = runs
            print(f"  Merged {agent_name} ({len(runs)} runs)")

    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nSaved → {raw_path}")
    print(f"Agents now: {[(k, len(v)) for k, v in raw.items()]}")

    summary = save_results(raw, EP_PER_TASK)
    print_table(summary)


if __name__ == "__main__":
    main()
