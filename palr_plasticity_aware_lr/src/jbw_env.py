"""
JellyBeanWorld Environment Wrapper for PALR Big-World Experiments
=================================================================
Wraps the JBW gym environment to create a continual RL benchmark that
captures the three core "big world" properties:

  1. Partial observability  — agent sees only a local visual patch
  2. Infinite horizon       — no natural episode boundary; episodes are
                              artificial windows of `steps_per_episode` steps
  3. Continuous non-stationarity — reward polarity flips every
                              `phase_episodes` episodes WITHOUT telling the
                              agent (no oracle), simulating the realistic
                              scenario where task boundaries are unannounced

Non-stationarity mechanism
--------------------------
JBW-v1 has two dominant item types: "jellybean" (positive reward) and
"onion" (negative reward). Every `phase_episodes` episodes we INVERT the
reward sign. The agent's optimal policy must therefore reverse — seek what
was previously avoided, and avoid what was previously sought.

This is stricter than Continual CartPole (physics change) because:
  - The reward function itself changes, not just the dynamics
  - There is NO task-switch oracle
  - The world state (item positions) is continuous across phase boundaries

Installation
------------
    pip install jelly-bean-world
    # Verify:
    python -c "import jbw; print(jbw.__version__)"

Usage
-----
    env = ContinualJBW(phase_episodes=50, steps_per_episode=500, seed=42)
    obs = env.reset()              # shape: (obs_dim,)
    obs, reward, done, info = env.step(action)   # action in {0,1,2,3}
    env.close()
"""

import numpy as np
import gym


# ---------------------------------------------------------------------------
# JBW availability check
# ---------------------------------------------------------------------------
try:
    import jbw  # noqa: F401 — registers JBW gym envs
    _JBW_AVAILABLE = True
except ImportError:
    _JBW_AVAILABLE = False


def _check_jbw():
    if not _JBW_AVAILABLE:
        raise ImportError(
            "jelly-bean-world is not installed.\n"
            "Run:  pip install jelly-bean-world\n"
            "Then: python -c \"import jbw; print(jbw.__version__)\""
        )


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
def _flatten_obs(obs) -> np.ndarray:
    """
    Convert JBW observation (dict or array) to a flat float32 vector.

    JBW-v0 returns a dict: {'vision': ndarray(11,11,3), 'scent': ndarray(3,), 'moved': scalar}
    Some versions return a flat array directly.
    """
    if isinstance(obs, dict):
        parts = []
        if "vision" in obs:
            parts.append(np.array(obs["vision"], dtype=np.float32).flatten())
        if "scent" in obs:
            parts.append(np.array(obs["scent"], dtype=np.float32).flatten())
        if "moved" in obs:
            parts.append(np.array([obs["moved"]], dtype=np.float32))
        return np.concatenate(parts)
    # Already a flat array
    return np.array(obs, dtype=np.float32).flatten()


def _get_vision_frame(obs) -> np.ndarray:
    """Extract the 11×11×3 vision array from an observation dict for video recording."""
    if isinstance(obs, dict) and "vision" in obs:
        arr = np.array(obs["vision"], dtype=np.float32)
        # Normalise to [0, 255] uint8
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
        return arr
    return None


# ---------------------------------------------------------------------------
# Environment IDs to try in order (JBW version-dependent)
# ---------------------------------------------------------------------------
_JBW_ENV_IDS = [
    "JBW-v0",
    "JBW-render-v0",
    "JBW-v1",
    "JBW-COMP579-obj-v1",
]


def _correct_reward_fn(prev_items, items):
    """
    Correct per-step reward: +1 for each jellybean collected, -1 for each onion.
    JBW-v0's built-in reward function is broken (uses len() instead of diff).
    Item indices: 0=banana(0), 1=onion(-1), 2=jellybean(+1), 3=wall(0)
    """
    prev = np.array(prev_items, dtype=np.int64)
    curr = np.array(items, dtype=np.int64)
    diff = curr - prev
    reward_weights = np.zeros(len(diff))
    if len(diff) >= 3:
        reward_weights[1] = -1.0  # onion
        reward_weights[2] = +1.0  # jellybean
    return float((diff * reward_weights).sum())


def _make_base_env(seed: int) -> gym.Env:
    """Try registered JBW env IDs until one succeeds; patch the reward function."""
    _check_jbw()
    last_err = None
    for env_id in _JBW_ENV_IDS:
        try:
            env = gym.make(env_id)
            # Patch the reward function on the underlying JBWEnv
            # (JBW-v0's default is broken: len(items)-len(prev_items) always = 0)
            inner = env.unwrapped
            if hasattr(inner, "_reward_fn"):
                inner._reward_fn = _correct_reward_fn
            return env
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not create any JBW gym environment.\n"
        f"Tried: {_JBW_ENV_IDS}\n"
        f"Last error: {last_err}\n\n"
        "Make sure jelly-bean-world is installed:\n"
        "    pip install jelly-bean-world"
    )


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class ContinualJBW:
    """
    JellyBeanWorld with periodic reward inversion — no oracle.

    The agent is NEVER told when phases change. It must detect distribution
    shift (if at all) purely from reward signals and its own internal state.

    Args:
        phase_episodes:    Episodes between reward inversions.
        steps_per_episode: Environment steps per artificial episode.
        seed:              Random seed.
    """

    def __init__(
        self,
        phase_episodes: int = 50,
        steps_per_episode: int = 500,
        seed: int = 42,
    ):
        self._base_env = _make_base_env(seed)
        self._base_env.seed(seed)
        np.random.seed(seed)

        self.phase_episodes    = phase_episodes
        self.steps_per_episode = steps_per_episode

        # One-time world initialisation (never destroyed between episodes)
        raw_obs = self._base_env.reset()
        self._obs_dim   = _flatten_obs(raw_obs).shape[0]
        self._n_actions = self._base_env.action_space.n  # 3 for JBW-v0

        # State
        self.episode_count       = 0
        self.phase_idx           = 0
        self._reward_sign        = +1.0   # flips every phase_episodes episodes
        self._step_in_episode    = 0
        self.task_switch_episodes: list = []

        self._last_raw_obs = raw_obs
        self._current_obs  = _flatten_obs(raw_obs)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Called at the start of each artificial episode.

        IMPORTANT: We do NOT call _base_env.reset() here. JBW is an infinite
        world — calling reset() on the gym env destroys the entire simulator and
        creates a new one, which is wrong for infinite-horizon continual RL.
        We simply reset the step counter and return the current observation.

        Phase flips are applied silently (no oracle to the agent).
        """
        # Phase boundary check (invisible to the agent)
        if (self.episode_count > 0 and
                self.episode_count % self.phase_episodes == 0):
            self.phase_idx   += 1
            self._reward_sign = -self._reward_sign
            self.task_switch_episodes.append(self.episode_count)

        self.episode_count    += 1
        self._step_in_episode  = 0
        return self._current_obs.copy()

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Take one environment step.
        Reward is multiplied by the current phase sign (unknown to agent).
        Episode ends after `steps_per_episode` steps.
        """
        raw_obs, raw_reward, _done, info = self._base_env.step(action)
        self._last_raw_obs   = raw_obs
        self._current_obs    = _flatten_obs(raw_obs)
        self._step_in_episode += 1

        reward = float(raw_reward) * self._reward_sign
        done   = (self._step_in_episode >= self.steps_per_episode)

        info["phase"]        = self.phase_idx
        info["reward_sign"]  = self._reward_sign
        return self._current_obs.copy(), reward, done, info

    # ------------------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def current_phase(self) -> int:
        return self.phase_idx

    @property
    def current_task(self) -> str:
        sign = "Normal" if self._reward_sign > 0 else "Inverted"
        return f"Phase-{self.phase_idx}-{sign}"

    def get_vision_frame(self) -> np.ndarray:
        """Return current vision frame (11×11×3 uint8) for video recording."""
        return _get_vision_frame(self._last_raw_obs)

    def close(self):
        self._base_env.close()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ContinualJBW...")
    env = ContinualJBW(phase_episodes=5, steps_per_episode=20, seed=42)
    print(f"  obs_dim  = {env.obs_dim}")
    print(f"  n_actions = {env.n_actions}")

    total_reward = 0.0
    for ep in range(12):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = np.random.randint(env.n_actions)
            obs, r, done, info = env.step(action)
            ep_reward += r
        total_reward += ep_reward
        print(f"  ep {ep:2d} | phase={info['phase']} "
              f"sign={info['reward_sign']:+.0f} | reward={ep_reward:.1f}")

    print(f"\nPhase switches at episodes: {env.task_switch_episodes}")
    print(f"Total reward over 12 eps: {total_reward:.1f}")
    env.close()
    print("OK")
