#!/usr/bin/env bash
# =============================================================================
#  palr_habitat/download_data.sh
#
#  Downloads the datasets required for Habitat Fetch Rearrangement:
#    1. ReplicaCAD  — photorealistic apartment scenes (articulated furniture)
#    2. YCB-Video   — manipulable objects (apple, bowl, etc.)
#    3. Rearrangement episode datasets
#
#  Uses habitat-lab's built-in download utility.
#  All data lands in  data/  inside this directory.
#
#  Usage:
#    conda activate palr_habitat
#    bash download_data.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

mkdir -p "${DATA_DIR}"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  PALR-Habitat data download"
echo "  Destination: ${DATA_DIR}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# 1. ReplicaCAD — scenes
# --------------------------------------------------------------------------- #
echo ""
echo "=== [1/3]  ReplicaCAD scenes (~4 GB) ==="
python -m habitat_sim.utils.datasets_download \
    --username "user@example.com" \
    --uids replica_cad_baked_lighting \
    --data-path "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 2. YCB objects
# --------------------------------------------------------------------------- #
echo ""
echo "=== [2/3]  YCB-Video objects (~1 GB) ==="
python -m habitat_sim.utils.datasets_download \
    --uids ycb \
    --data-path "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 3. Rearrangement episode datasets
#    These define which objects to pick, receptacle targets, etc.
# --------------------------------------------------------------------------- #
echo ""
echo "=== [3/3]  Rearrangement episode datasets ==="
python -m habitat_sim.utils.datasets_download \
    --uids rearrange_pick_dataset_v0 \
    --data-path "${DATA_DIR}"

echo ""
echo "============================================================"
echo "  Data download complete."
echo "  Directory layout:"
echo "    ${DATA_DIR}/"
echo "    ├── ReplicaCAD/"
echo "    ├── objects/ycb/"
echo "    └── datasets/rearrange/"
echo ""
echo "  Next step:  bash launch.sh --agent palr --seeds 5"
echo "============================================================"
