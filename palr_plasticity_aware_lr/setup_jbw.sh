#!/usr/bin/env bash
# =============================================================================
# setup_jbw.sh — Install and verify JellyBeanWorld for PALR experiments
# =============================================================================
# Usage:
#   chmod +x setup_jbw.sh
#   ./setup_jbw.sh
#
# After this script succeeds, run experiments with:
#   cd src/
#   python run_jbw_experiments.py --fast          # quick debug (1 seed, 80 eps)
#   python run_jbw_experiments.py                 # full run (3 seeds, 200 eps)
#   python plot_jbw_results.py                    # generate figures
# =============================================================================

set -e   # exit on any error

echo "============================================="
echo " PALR + JellyBeanWorld Setup"
echo "============================================="

# ---------------------------------------------------------------------------
# 1. Core dependencies (should already be installed for CartPole experiments)
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Checking core dependencies..."
python3 -c "import numpy, tensorflow, gym, matplotlib" \
    && echo "  ✓ numpy, tensorflow, gym, matplotlib" \
    || { echo "  ✗ Missing core deps. Run: pip install numpy tensorflow gym matplotlib"; exit 1; }

# ---------------------------------------------------------------------------
# 2. Install jelly-bean-world
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Installing jelly-bean-world..."

if python3 -c "import jbw" 2>/dev/null; then
    JBW_VERSION=$(python3 -c "import jbw; print(jbw.__version__)" 2>/dev/null || echo "unknown")
    echo "  ✓ jelly-bean-world already installed (version: $JBW_VERSION)"
else
    echo "  Installing via pip..."
    pip install jelly-bean-world
    echo "  ✓ jelly-bean-world installed"
fi

# ---------------------------------------------------------------------------
# 3. Smoke test — create env, take a few steps, confirm obs/action shapes
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Running smoke test..."

python3 - <<'PYEOF'
import sys
sys.path.insert(0, "src")

try:
    from jbw_env import ContinualJBW
    import numpy as np

    env = ContinualJBW(phase_episodes=3, steps_per_episode=10, seed=42)
    print(f"  obs_dim   = {env.obs_dim}")
    print(f"  n_actions = {env.n_actions}")

    rewards = []
    for ep in range(7):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,), f"Bad obs shape: {obs.shape}"
        done = False
        ep_r = 0
        while not done:
            action = np.random.randint(env.n_actions)
            obs, r, done, info = env.step(action)
            ep_r += r
        rewards.append(ep_r)
        print(f"  ep {ep}: phase={info['phase']} sign={info['reward_sign']:+.0f} reward={ep_r:.2f}")

    print(f"\n  Phase switches recorded: {env.task_switch_episodes}")
    env.close()
    print("\n  ✓ Smoke test passed")

except ImportError as e:
    print(f"\n  ✗ Import error: {e}")
    print("  Try: pip install jelly-bean-world")
    sys.exit(1)
except Exception as e:
    print(f"\n  ✗ Error: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup complete!"
echo "============================================="
echo ""
echo " Quick debug run (1 seed, 80 episodes, ~5 min):"
echo "   cd src && python run_jbw_experiments.py --fast"
echo ""
echo " Full run (3 seeds, 200 episodes per agent, ~30-60 min):"
echo "   cd src && python run_jbw_experiments.py"
echo ""
echo " Generate figures after run:"
echo "   cd src && python plot_jbw_results.py"
echo ""
echo " Recommended full experiment for paper:"
echo "   cd src && python run_jbw_experiments.py \\"
echo "       --episodes 400 --seeds 5 --phase_episodes 100 --steps_per_ep 500"
echo "============================================="
