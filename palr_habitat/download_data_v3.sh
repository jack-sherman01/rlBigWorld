#!/usr/bin/env bash
# =============================================================================
#  palr_habitat/download_data_v3.sh
#
#  Downloads datasets compatible with habitat-sim/lab 0.3.2.
#  ReplicaCAD rearrange episode JSONs were re-issued in 0.3.x with extra
#  schema fields; the 0.2.5 dataset is NOT loadable in 0.3.2.
#
#  All files land in  data/  inside this directory.
#
#  Usage:
#    conda activate palr_habitat_v3
#    bash download_data_v3.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"

mkdir -p "${DATA_DIR}"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  PALR-Habitat v3 data download (habitat 0.3.2)"
echo "  Destination: ${DATA_DIR}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# 1. ReplicaCAD baked-lighting scenes (~4 GB)
# --------------------------------------------------------------------------- #
echo ""
echo "=== [1/4]  ReplicaCAD scenes ==="
python -m habitat_sim.utils.datasets_download \
    --uids replica_cad_baked_lighting \
    --data-path "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 2. YCB objects (~1 GB)
# --------------------------------------------------------------------------- #
echo ""
echo "=== [2/4]  YCB objects ==="
python -m habitat_sim.utils.datasets_download \
    --uids ycb \
    --data-path "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 3. Hab-Fetch robot URDF
# --------------------------------------------------------------------------- #
echo ""
echo "=== [3/4]  Hab-Fetch robot ==="
python -m habitat_sim.utils.datasets_download \
    --uids hab_fetch \
    --data-path "${DATA_DIR}"

# --------------------------------------------------------------------------- #
# 4. Rearrangement episode datasets (0.3.x schema)
# --------------------------------------------------------------------------- #
echo ""
echo "=== [4/4]  Rearrangement episode datasets ==="
python -m habitat_sim.utils.datasets_download \
    --uids rearrange_pick_dataset_v0 rearrange_dataset_v1 \
    --data-path "${DATA_DIR}"

echo ""
echo "============================================================"
echo "  Data download complete."
echo ""
echo "  Quick check:"
echo "    ls ${DATA_DIR}/datasets/replica_cad/rearrange/v1/train/"
echo "============================================================"
