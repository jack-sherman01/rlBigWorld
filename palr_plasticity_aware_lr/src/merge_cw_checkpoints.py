"""
Merge per-agent-per-seed checkpoint files into final results.

Usage:
    bash run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py           # CW10
    bash run.sh palr_plasticity_aware_lr/src/merge_cw_checkpoints.py --suffix cw20  # CW20
"""
import os, sys, json, glob, argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="",
                        help="Checkpoint suffix to scan (e.g. 'cw20'). "
                             "Scans cw_checkpoint_{suffix}_seed*_agent*.json")
    args = parser.parse_args()
    ckpt_prefix = f"cw_checkpoint_{args.suffix}" if args.suffix else "cw_checkpoint"
    out_name    = f"{args.suffix}_raw_results.json" if args.suffix else "cw_raw_results.json"
    merged = {}
    loaded = []

    # New naming: one file per agent per seed
    pattern_new = os.path.join(RESULTS_DIR, f"{ckpt_prefix}_seed*_agent*.json")
    new_files = sorted(glob.glob(pattern_new))

    # Legacy naming (CW10 only, no suffix case)
    pattern_old = os.path.join(RESULTS_DIR, "cw_checkpoint_seed[0-9].json")
    old_files = sorted(glob.glob(pattern_old)) if not args.suffix else []

    # Determine which (seed, agent) pairs are already covered by new-style files
    covered_pairs = set()
    for f in new_files:
        base = os.path.basename(f)
        try:
            seed_str  = base.split("_seed")[1].split("_")[0]
            agent_str = base.split("_agent")[1].replace(".json", "")
            covered_pairs.add((int(seed_str), int(agent_str)))
        except (IndexError, ValueError):
            pass

    # Always include old-style seed files — they may hold agents not re-run
    # (e.g. a4/a6 saved before the race-condition was fixed).
    # Only skip an old file if every agent in it is already covered by a new file.
    files_to_load = new_files[:]
    for f in old_files:
        with open(f) as fh:
            data = json.load(fh)
        base = os.path.basename(f)
        try:
            seed_idx = int(base.replace("cw_checkpoint_seed", "").replace(".json", ""))
        except ValueError:
            continue
        # Map agent name back to agent index
        name_to_idx = {
            "SAC-FixedLR": 0, "SAC-ShrinkAndPerturb": 1, "SAC-PeriodicReset": 2,
            "SAC-L2Reg": 3, "PALR-SAC (ours)": 4,
            "PALR-SAC-NoPerturb": 5, "PALR-SAC-NoScale": 6,
        }
        already_covered = all(
            (seed_idx, name_to_idx.get(agent, -1)) in covered_pairs
            for agent in data.keys()
        )
        if not already_covered:
            files_to_load.append(f)

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

    raw_path = os.path.join(RESULTS_DIR, out_name)
    with open(raw_path, "w") as f:
        json.dump(to_serialisable(merged), f, indent=2)
    print(f"\nMerged {len(files_to_load)} files → {raw_path}")
    print(f"Agents: {list(merged.keys())}")
    print(f"Runs per agent: {[len(v) for v in merged.values()]}")

if __name__ == "__main__":
    main()
