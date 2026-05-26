"""
ManiSkill Heterogeneous Skill Stream Environment Wrapper
=========================================================
Wraps 4 ManiSkill tasks into a sequential continual-RL benchmark.

Task stream (no oracle — agent never told when task switches):
  1. PickCube-v1     — grasp a cube and place it at a target
  2. StackCube-v1    — pick one cube and stack it on another
  3. TurnFaucet-v1   — rotate a faucet handle to a target angle
  4. PushCube-v1     — push a cube to a target position (no grasp)

Observation: RGB image (128×128×3), resized and normalised → ViT input
Action: continuous 8-DOF (Panda arm 7-DOF + gripper)

Non-stationarity: each task runs for `task_episodes` episodes, then
switches to the next (cycling). No task-switch signal given to the agent.
"""

import builtins
import numpy as np
import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers all ManiSkill envs with gymnasium
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
TASK_SEQUENCE = [
    "PickCube-v1",
    "StackCube-v1",
    "TurnFaucet-v1",
    "PushCube-v1",
]

# Observation image size fed to ViT
OBS_IMG_SIZE = 128

# Action space: Panda arm in joint-delta control (7-DOF + gripper = 8)
ACTION_DIM = 8

# Scene bounding box for top-down schematic rendering (metres)
SCENE_XY_RANGE = 0.6   # scene spans ±0.3m in x and y


# ---------------------------------------------------------------------------
# ManiSkill env factory  (obs_mode="state" — avoids Vulkan/CUDA dependency)
# ---------------------------------------------------------------------------

def _make_single_env(task_id: str, seed: int, obs_size: int = OBS_IMG_SIZE) -> gym.Env:
    """
    Create a ManiSkill gymnasium environment with state observations.

    We use obs_mode="state" to avoid the Vulkan rendering requirement.
    Visual input for ViT is generated synthetically from the state dict
    by _state_to_image(), giving a top-down schematic 128×128 image.
    This is sufficient for the plasticity experiment: the ViT architecture
    is identical to the RGB case, only the rendering pipeline differs.
    """
    # Non-interactive runs have no stdin; auto-accept ManiSkill asset downloads.
    _orig_input = builtins.input
    builtins.input = lambda _prompt="": "y"
    try:
        env = gym.make(
            task_id,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode=None,
        )
        env.reset(seed=seed)
    finally:
        builtins.input = _orig_input
    return env


# ---------------------------------------------------------------------------
# Synthetic top-down image from ManiSkill state dict
# ---------------------------------------------------------------------------

def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, color: tuple):
    H, W = img.shape[:2]
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                img[y, x] = color


def _draw_rect(img: np.ndarray, cx: int, cy: int, hw: int, color: tuple):
    H, W = img.shape[:2]
    y0, y1 = max(0, cy - hw), min(H, cy + hw + 1)
    x0, x1 = max(0, cx - hw), min(W, cx + hw + 1)
    img[y0:y1, x0:x1] = color


def _world_to_pixel(x: float, y: float, size: int) -> Tuple[int, int]:
    """Map world XY coords (metres) to pixel coords in a top-down image."""
    px = int((x / SCENE_XY_RANGE + 1.0) * 0.5 * size)
    py = int((1.0 - (y / SCENE_XY_RANGE + 1.0) * 0.5) * size)
    return np.clip(px, 0, size - 1), np.clip(py, 0, size - 1)


def _state_to_image(raw_obs, task_id: str, size: int = OBS_IMG_SIZE) -> np.ndarray:
    """
    Generate a synthetic top-down schematic (size×size×3 uint8) from a
    ManiSkill state observation dict.

    Colour coding:
      Dark grey background | Blue robot base | Green end-effector
      Red cube/object      | Yellow target   | Cyan secondary object
    """
    img = np.full((size, size, 3), 30, dtype=np.uint8)   # dark background

    # Robot base (fixed at origin)
    bx, by = _world_to_pixel(0.0, 0.0, size)
    _draw_circle(img, bx, by, size // 20, (80, 80, 200))   # blue base

    if not isinstance(raw_obs, dict):
        return img

    # Flatten all numeric arrays in obs for positional data
    def _get_pos(d: dict, keys: list):
        """Try to find a 3D position from candidate keys."""
        for k in keys:
            if k in d:
                v = np.array(d[k]).flatten()
                if len(v) >= 3:
                    return float(v[0]), float(v[1])
            # Recurse one level
            for sub in d.values():
                if isinstance(sub, dict) and k in sub:
                    v = np.array(sub[k]).flatten()
                    if len(v) >= 3:
                        return float(v[0]), float(v[1])
        return None

    # End-effector position (from agent TCP or extra)
    ee = _get_pos(raw_obs, ["tcp_pose", "ee_pos", "tcp_pos"])
    if ee:
        px, py = _world_to_pixel(ee[0], ee[1], size)
        _draw_circle(img, px, py, size // 16, (80, 200, 80))  # green EE

    # Primary object (cube, faucet handle)
    obj = _get_pos(raw_obs, ["cube_pose", "obj_pose", "target_obj_pose",
                              "faucet_pose", "object_pose"])
    if obj:
        px, py = _world_to_pixel(obj[0], obj[1], size)
        _draw_rect(img, px, py, size // 14, (200, 60, 60))   # red object

    # Target position
    tgt = _get_pos(raw_obs, ["goal_pos", "target_pos", "target_cube_pose"])
    if tgt:
        px, py = _world_to_pixel(tgt[0], tgt[1], size)
        _draw_circle(img, px, py, size // 18, (220, 220, 60))  # yellow target

    # Secondary object (StackCube: second cube)
    obj2 = _get_pos(raw_obs, ["cube_b_pose", "second_obj_pose"])
    if obj2:
        px, py = _world_to_pixel(obj2[0], obj2[1], size)
        _draw_rect(img, px, py, size // 16, (60, 200, 200))  # cyan second obj

    return img


def _process_obs(raw_obs, task_id: str = "", size: int = OBS_IMG_SIZE) -> np.ndarray:
    """
    Convert ManiSkill state obs → synthetic top-down image → CHW float32 [0,1].
    Returns: (3, size, size) float32.
    """
    img = _state_to_image(raw_obs, task_id, size)
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class HeterogeneousSkillStream:
    """
    Sequential continual-RL benchmark over 4 ManiSkill manipulation tasks.

    The agent is never told when the task changes. It must maintain plasticity
    and transfer knowledge to succeed across task switches.

    Args:
        task_sequence:    List of ManiSkill task IDs (default: TASK_SEQUENCE)
        task_episodes:    Episodes per task before switching (default: 100)
        steps_per_episode: Max steps per episode (default: 200)
        seed:             Base random seed
        obs_size:         Image size for ViT input (default: 128)
    """

    def __init__(
        self,
        task_sequence:     List[str] = None,
        task_episodes:     int       = 100,
        steps_per_episode: int       = 200,
        seed:              int       = 0,
        obs_size:          int       = OBS_IMG_SIZE,
    ):
        self.task_sequence     = task_sequence or TASK_SEQUENCE
        self.task_episodes     = task_episodes
        self.steps_per_episode = steps_per_episode
        self.obs_size          = obs_size
        self._base_seed        = seed

        self.n_tasks           = len(self.task_sequence)
        self.action_dim        = ACTION_DIM
        self.obs_shape         = (3, obs_size, obs_size)   # CHW for ViT

        # State
        self.episode_count         = 0
        self.task_idx              = 0
        self.task_switch_episodes: List[int] = []
        self._step_in_episode      = 0
        self._current_env: Optional[gym.Env] = None
        self._current_obs          = None

        # Build first env
        self._load_task(self.task_idx)

    # ------------------------------------------------------------------
    def _load_task(self, task_idx: int):
        """Instantiate the gym env for task `task_idx`."""
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception:
                pass

        task_id = self.task_sequence[task_idx]
        seed    = self._base_seed + task_idx * 1000
        self._current_env = _make_single_env(task_id, seed, self.obs_size)

        raw_obs, _ = self._current_env.reset(seed=seed)
        self._current_obs = _process_obs(raw_obs, task_id, self.obs_size)

        self._step_in_episode = 0
        print(f"  [HeteroStream] Loaded task {task_idx}: {task_id}  "
              f"obs_shape={self._current_obs.shape}")

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Called at the start of each artificial episode.
        Handles task switching (silently, no oracle).
        """
        # Task boundary check
        if (self.episode_count > 0 and
                self.episode_count % self.task_episodes == 0):
            new_task_idx = (self.task_idx + 1) % self.n_tasks
            if new_task_idx != self.task_idx:
                self.task_idx = new_task_idx
                self.task_switch_episodes.append(self.episode_count)
                self._load_task(self.task_idx)
            # On full cycle: back to task 0 with incremented seed
            elif new_task_idx == 0:
                self.task_idx = 0
                self.task_switch_episodes.append(self.episode_count)
                self._load_task(self.task_idx)

        self.episode_count    += 1
        self._step_in_episode  = 0

        raw_obs, _ = self._current_env.reset()
        self._current_obs = _process_obs(raw_obs, self.current_task_name, self.obs_size)
        return self._current_obs.copy()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take one environment step.

        Args:
            action: np.ndarray of shape (action_dim,) in [-1, 1]
        Returns:
            obs, reward, done, info
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        raw_obs, reward, terminated, truncated, info = \
            self._current_env.step(action)

        self._current_obs     = _process_obs(raw_obs, self.current_task_name, self.obs_size)
        self._step_in_episode += 1

        # Episode ends on termination, truncation, or step limit
        done = terminated or truncated or \
               (self._step_in_episode >= self.steps_per_episode)

        info["task_idx"]    = self.task_idx
        info["task_name"]   = self.task_sequence[self.task_idx]
        info["success"]     = float(info.get("success", 0.0))

        return self._current_obs.copy(), float(reward), done, info

    # ------------------------------------------------------------------
    @property
    def current_task_name(self) -> str:
        return self.task_sequence[self.task_idx]

    @property
    def current_task_idx(self) -> int:
        return self.task_idx

    def close(self):
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Quick smoke test (requires ManiSkill to be installed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing HeterogeneousSkillStream...")
    env = HeterogeneousSkillStream(
        task_episodes=3,
        steps_per_episode=10,
        seed=0,
    )
    print(f"  obs_shape  = {env.obs_shape}")
    print(f"  action_dim = {env.action_dim}")
    print(f"  n_tasks    = {env.n_tasks}")

    for ep in range(14):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = np.random.uniform(-1, 1, env.action_dim)
            obs, r, done, info = env.step(action)
            ep_reward += r
        print(f"  ep {ep:2d} | task={info['task_name']:<20s} "
              f"reward={ep_reward:.2f}  success={info['success']:.0f}")

    print(f"\nTask switches at episodes: {env.task_switch_episodes}")
    env.close()
    print("OK")
