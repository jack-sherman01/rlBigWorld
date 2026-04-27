#!/usr/bin/env python3
"""
Download Habitat datasets from Hugging Face (no conda/habitat_sim required).

Usage:
    python3 download_data_hf.py
"""
import os
from huggingface_hub import snapshot_download

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

datasets = [
    # (repo_id, local_subdir)
    ("ai-habitat/ReplicaCAD_baked_lighting", "replica_cad_baked_lighting"),
    ("ai-habitat/ycb",                       "objects/ycb"),
]

for repo_id, subdir in datasets:
    local_dir = os.path.join(DATA_DIR, subdir)
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*60}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print(f"Done: {repo_id}")

print("\n" + "="*60)
print("All downloads complete.")
print(f"Data directory: {DATA_DIR}")
