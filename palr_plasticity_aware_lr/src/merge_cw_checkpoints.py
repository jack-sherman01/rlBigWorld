"""
Merge per-seed CW10 checkpoint files into final results.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py
"""
import os, sys, json
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
    for seed_idx in range(5):
        path = os.path.join(RESULTS_DIR, f"cw_checkpoint_seed{seed_idx}.json")
        if not os.path.exists(path):
            print(f"  Missing: {path} — skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for agent, runs in data.items():
            merged.setdefault(agent, []).extend(runs)
        print(f"  Loaded seed{seed_idx}: {[f'{k}={len(v)}runs' for k,v in data.items()]}")

    raw_path = os.path.join(RESULTS_DIR, "cw_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(to_serialisable(merged), f, indent=2)
    print(f"\nMerged → {raw_path}")
    print(f"Agents: {list(merged.keys())}")
    print(f"Runs per agent: {[len(v) for v in merged.values()]}")

if __name__ == "__main__":
    main()
