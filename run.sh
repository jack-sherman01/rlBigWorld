#!/usr/bin/env bash
# Launcher for PALR experiments — sets GPU env and activates venv
VENV="$(dirname "$0")/.venv"
NVIDIA_LIBS=$(find "$VENV/lib/python3.10/site-packages/nvidia" -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl        # headless MuJoCo rendering (CW10 experiments)
"$VENV/bin/python" "$@"
