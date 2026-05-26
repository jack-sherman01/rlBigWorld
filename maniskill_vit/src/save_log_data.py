"""
save_log_data.py — Extract and save all training data from logs + checkpoints.

Outputs to maniskill_vit/results/data/:
  training_log.csv      — from agent*_seed*.log (all 18 runs, every 10 eps)
  episode_data.csv      — from checkpoint JSONs (per-episode rewards/successes)
  plasticity_data.csv   — from checkpoint JSONs (per-layer dead/erank every 5 eps)
  palr_history.csv      — from checkpoint JSONs (PALR scale history)

Run from the project root:
  python maniskill_vit/src/save_log_data.py
"""

import csv
import json
import math
import os
import re

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR  = os.path.join(PROJ_DIR, "logs")
CKPT_DIR = os.path.join(PROJ_DIR, "maniskill_vit", "results")
OUT_DIR  = os.path.join(PROJ_DIR, "maniskill_vit", "results", "data")

AGENT_NAMES = {
    0: "SAC-FixedLR",
    1: "SAC-L2Reg",
    2: "SAC-ShrinkPerturb",
    3: "PALR-SAC",
    4: "PALR-NoPerturb",
    5: "PALR-NoScale",
}

# Regex for log lines:  ep    0/400 | task=PickCube-v1 | r=  2.27 | dead_L5=nan | 1s
EP_RE = re.compile(
    r"ep\s+(\d+)/\d+\s+\|\s+task=(\S+?)\s+\|\s+r=\s*([\d.\-]+)\s+\|\s+dead_L5=([\d.nan]+)\s+\|\s+(\d+)s"
)


def parse_log(path: str, agent_idx: int, seed: int) -> list[dict]:
    agent_name = AGENT_NAMES[agent_idx]
    rows = []
    with open(path) as f:
        for line in f:
            m = EP_RE.search(line)
            if m:
                ep, task, r, dead, t = m.groups()
                rows.append({
                    "agent_idx":  agent_idx,
                    "agent_name": agent_name,
                    "seed":       seed,
                    "episode":    int(ep),
                    "task":       task,
                    "reward":     float(r),
                    "dead_L5":    float("nan") if dead == "nan" else float(dead),
                    "time_s":     int(t),
                })
    return rows


def parse_checkpoint(path: str) -> tuple[list, list, list]:
    """Returns (episode_rows, plasticity_rows, palr_rows)."""
    with open(path) as f:
        data = json.load(f)

    agent_name = list(data.keys())[0]
    run = data[agent_name][0]
    agent_idx  = run["agent_idx"]
    seed       = run["seed"]

    task_seq = ["PickCube-v1", "StackCube-v1", "TurnFaucet-v1", "PushCube-v1"]

    # ── Episode rows ──────────────────────────────────────────────────────────
    ep_rows = []
    rewards   = run.get("episode_rewards", [])
    successes = run.get("episode_successes", [])
    task_ids  = run.get("episode_task_ids", [])
    for i, (r, s, tid) in enumerate(zip(rewards, successes, task_ids)):
        ep_rows.append({
            "agent_idx":  agent_idx,
            "agent_name": agent_name,
            "seed":       seed,
            "episode":    i,
            "task_id":    tid,
            "task":       task_seq[tid] if tid < len(task_seq) else str(tid),
            "reward":     r,
            "success":    s,
        })

    # ── Plasticity rows ───────────────────────────────────────────────────────
    plast_rows = []
    for entry in run.get("plasticity_log", []):
        base = {
            "agent_idx":  agent_idx,
            "agent_name": agent_name,
            "seed":       seed,
            "episode":    entry["episode"],
            "task":       entry.get("task", ""),
        }
        for key, val in entry.items():
            if key in ("episode", "task"):
                continue
            v = val if not (isinstance(val, float) and math.isnan(val)) else None
            base[key] = v
        plast_rows.append(base)

    # ── PALR history rows ─────────────────────────────────────────────────────
    palr_rows = []
    for entry in run.get("palr_history", []):
        base = {
            "agent_idx":  agent_idx,
            "agent_name": agent_name,
            "seed":       seed,
            "step":       entry["step"],
        }
        dead   = entry.get("dead", [])
        erank  = entry.get("erank", [])
        scales = entry.get("scales", [])
        for i, v in enumerate(dead):
            base[f"dead_L{i}"] = v
        for i, v in enumerate(erank):
            base[f"erank_L{i}"] = v
        for i, v in enumerate(scales):
            base[f"scale_L{i}"] = v
        palr_rows.append(base)

    return ep_rows, plast_rows, palr_rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        print(f"  (no data for {os.path.basename(path)})")
        return
    keys = list(rows[0].keys())
    # Collect all keys in case later rows have extra columns
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {len(rows):>5} rows → {os.path.relpath(path, PROJ_DIR)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Parse all log files ────────────────────────────────────────────────
    print("Parsing log files …")
    log_rows = []
    for agent_idx in range(6):
        for seed in range(3):
            fname = f"agent{agent_idx}_seed{seed}.log"
            path  = os.path.join(LOG_DIR, fname)
            if not os.path.exists(path):
                print(f"  missing: {fname}")
                continue
            rows = parse_log(path, agent_idx, seed)
            print(f"  {fname}: {len(rows)} ep entries")
            log_rows.extend(rows)

    write_csv(os.path.join(OUT_DIR, "training_log.csv"), log_rows)

    # ── 2. Parse checkpoint JSONs ─────────────────────────────────────────────
    print("\nParsing checkpoint JSONs …")
    all_ep, all_plast, all_palr = [], [], []

    for fname in sorted(os.listdir(CKPT_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CKPT_DIR, fname)
        try:
            ep_rows, plast_rows, palr_rows = parse_checkpoint(path)
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue
        print(f"  {fname}: {len(ep_rows)} episodes, {len(plast_rows)} plasticity entries, {len(palr_rows)} palr entries")
        all_ep.extend(ep_rows)
        all_plast.extend(plast_rows)
        all_palr.extend(palr_rows)

    write_csv(os.path.join(OUT_DIR, "episode_data.csv"),   all_ep)
    write_csv(os.path.join(OUT_DIR, "plasticity_data.csv"), all_plast)
    write_csv(os.path.join(OUT_DIR, "palr_history.csv"),    all_palr)

    print("\nDone.")


if __name__ == "__main__":
    main()
