"""
Merge per-agent-per-seed CW10 checkpoint files into final results.

Scans for all cw_checkpoint_seed{N}_agent{M}.json files in the results
directory and merges them.  Falls back to the older cw_checkpoint_seed{N}.json
naming for any files that exist with the previous convention.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def to_serialisable(obj):
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, dict):           return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [to_serialisable(i) for i in obj]
    return obj

def main():
    merged = {}
    loaded = []

    # New naming: one file per agent per seed (no race conditions)
    pattern_new = os.path.join(RESULTS_DIR, "cw_checkpoint_seed*_agent*.json")
    new_files = sorted(glob.glob(pattern_new))

    # Legacy naming: one file per seed (may have partial results due to race cond)
    pattern_old = os.path.join(RESULTS_DIR, "cw_checkpoint_seed[0-9].json")
    old_files = sorted(glob.glob(pattern_old))

    # Determine which old seed files are NOT already covered by new-style files
    covered_seeds = set()
    for f in new_files:
        base = os.path.basename(f)
        # extract seed index from cw_checkpoint_seed{N}_agent{M}.json
        try:
            seed_str = base.split("_seed")[1].split("_")[0]
            covered_seeds.add(int(seed_str))
        except (IndexError, ValueError):
            pass

    files_to_load = new_files[:]
    for f in old_files:
        base = os.path.basename(f)
        try:
            seed_str = base.replace("cw_checkpoint_seed", "").replace(".json", "")
            if int(seed_str) not in covered_seeds:
                files_to_load.append(f)
        except ValueError:
            pass

    if not files_to_load:
        print("No checkpoint files found.")
        return

    for path in files_to_load:
        with open(path) as f:
            data = json.load(f)
        for agent, runs in data.items():
            merged.setdefault(agent, []).extend(runs)
        loaded.append(f"  {os.path.basename(path)}: {[f'{k}={len(v)}' for k,v in data.items()]}")

    for line in loaded:
        print(line)

    raw_path = os.path.join(RESULTS_DIR, "cw_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(to_serialisable(merged), f, indent=2)
    print(f"\nMerged {len(files_to_load)} files → {raw_path}")
    print(f"Agents: {list(merged.keys())}")
    print(f"Runs per agent: {[len(v) for v in merged.values()]}")

if __name__ == "__main__":
    main()
